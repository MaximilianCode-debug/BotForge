import os
import sqlite3
import threading
import asyncio
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import groq
from dotenv import load_dotenv
import hashlib

load_dotenv()

# ---------- CONFIG ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables")

DB_PATH = "database.db"

# ---------- DATABASE HELPERS ----------
def get_db():
    """Get database connection and ensure tables exist (for Render's ephemeral storage)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Create tables if they don't exist – runs on every request
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS businesses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_name TEXT NOT NULL,
                bot_token TEXT UNIQUE NOT NULL,
                business_context TEXT NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL,
                chat_id TEXT NOT NULL,
                user_message TEXT,
                bot_response TEXT,
                direction TEXT CHECK(direction IN ('incoming', 'outgoing')),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (business_id) REFERENCES businesses(id)
            )
        ''')
    return conn

# ---------- DB OPERATIONS ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_business(name, token, context, password):
    with get_db() as conn:
        hashed = hash_password(password)
        cur = conn.execute(
            "INSERT INTO businesses (business_name, bot_token, business_context, password) VALUES (?, ?, ?, ?)",
            (name, token, context, hashed)
        )
        business_id = cur.lastrowid
        conn.commit()
        return business_id

def get_business(business_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM businesses WHERE id = ?", (business_id,)).fetchone()
        return dict(row) if row else None

def get_business_by_token(token):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM businesses WHERE bot_token = ?", (token,)).fetchone()
        return dict(row) if row else None

def get_business_by_credentials(token, password):
    with get_db() as conn:
        hashed = hash_password(password)
        row = conn.execute(
            "SELECT * FROM businesses WHERE bot_token = ? AND password = ?",
            (token, hashed)
        ).fetchone()
        return dict(row) if row else None

def update_business(business_id, name=None, context=None, active=None, password=None):
    updates = []
    params = []
    if name is not None:
        updates.append("business_name = ?")
        params.append(name)
    if context is not None:
        updates.append("business_context = ?")
        params.append(context)
    if active is not None:
        updates.append("is_active = ?")
        params.append(1 if active else 0)
    if password is not None:
        updates.append("password = ?")
        params.append(hash_password(password))
    if not updates:
        return
    params.append(business_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE businesses SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            params
        )
        conn.commit()

def log_message(business_id, chat_id, user_msg=None, bot_resp=None, direction='incoming'):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (business_id, chat_id, user_message, bot_response, direction) VALUES (?, ?, ?, ?, ?)",
            (business_id, str(chat_id), user_msg, bot_resp, direction)
        )
        conn.commit()

def get_messages(business_id, limit=100):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE business_id = ? ORDER BY timestamp DESC LIMIT ?",
            (business_id, limit)
        ).fetchall()
        return [dict(row) for row in rows]

def get_analytics(business_id):
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM messages WHERE business_id = ?", (business_id,)).fetchone()[0]
        users = conn.execute("SELECT COUNT(DISTINCT chat_id) FROM messages WHERE business_id = ?", (business_id,)).fetchone()[0]
        yesterday = datetime.now() - timedelta(days=1)
        recent = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE business_id = ? AND timestamp >= ?",
            (business_id, yesterday)
        ).fetchone()[0]
        return {
            'total_messages': total,
            'unique_users': users,
            'messages_last_24h': recent
        }

# ---------- CONVERSATION HISTORY ----------
def get_conversation_history(business_id, chat_id, limit=5):
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT direction, user_message, bot_response
            FROM messages
            WHERE business_id = ? AND chat_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (business_id, str(chat_id), limit * 2)
        ).fetchall()
    rows = list(reversed(rows))
    history = []
    for row in rows:
        if row['direction'] == 'incoming':
            history.append({'role': 'user', 'content': row['user_message']})
        elif row['direction'] == 'outgoing':
            history.append({'role': 'assistant', 'content': row['bot_response']})
    return history

# ---------- GROQ ----------
groq_client = groq.Groq(api_key=GROQ_API_KEY)

async def ask_groq(messages: list, business_context: str) -> str:
    system_prompt = (
        f"You are a professional business assistant for {business_context}. "
        "Use the conversation history to provide coherent and context-aware answers. "
        "Be polite, concise, and helpful."
    )
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=full_messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return "⚠️ AI service temporarily unavailable. Please try later."

# ---------- TELEGRAM BOT ----------
running_bots = {}

def create_bot_app(token, business_id):
    business = get_business(business_id)
    if not business or not business['is_active']:
        return None

    app = Application.builder().token(token).build()

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            f"👋 *Hello! I'm the AI Assistant for {business['business_name']}.*\n\n"
            "I'm here to help you with:\n"
            "• Questions about our products and services\n"
            "• Business hours and location\n"
            "• Pricing and availability\n"
            "• Any other business-related inquiries\n\n"
            "Just ask me anything – I'll do my best to help!"
        )

    async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = str(update.effective_chat.id)
        with get_db() as conn:
            conn.execute(
                "DELETE FROM messages WHERE business_id = ? AND chat_id = ?",
                (business_id, chat_id)
            )
            conn.commit()
        await update.message.reply_text("🔄 Conversation history reset. Starting fresh!")

    async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = str(update.effective_chat.id)
        user_msg = update.message.text

        log_message(business_id, chat_id, user_msg=user_msg, direction='incoming')

        context_text = business['business_context']
        if not context_text:
            await update.message.reply_text("⚠️ Business info missing. Contact owner.")
            return

        history = get_conversation_history(business_id, chat_id, limit=5)
        messages = []
        for entry in history:
            messages.append({"role": entry['role'], "content": entry['content']})
        messages.append({"role": "user", "content": user_msg})

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        reply = await ask_groq(messages, context_text)

        log_message(business_id, chat_id, bot_resp=reply, direction='outgoing')
        await update.message.reply_text(reply)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))
    return app

def start_bot_thread(token, business_id):
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = create_bot_app(token, business_id)
        if app:
            running_bots[token] = app
            print(f"🤖 Bot for business {business_id} started polling...")
            app.run_polling(allowed_updates=Update.ALL_TYPES)
        else:
            print(f"❌ Failed to start bot for business {business_id}")
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread

# ---------- FLASK APP ----------
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ---------- ROUTES ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        token = request.form.get('token')
        password = request.form.get('password')
        if not token or not password:
            return render_template('login.html', error='Please enter both token and password.')
        business = get_business_by_credentials(token, password)
        if business:
            session['business_id'] = business['id']
            session['business_name'] = business['business_name']
            return redirect(url_for('dashboard', business_id=business['id']))
        else:
            return render_template('login.html', error='Invalid token or password.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard/<int:business_id>')
def dashboard(business_id):
    if 'business_id' not in session or session['business_id'] != business_id:
        return redirect(url_for('login'))
    business = get_business(business_id)
    if not business:
        return "Business not found", 404
    return render_template('dashboard.html', business=business)

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.get_json()
        name = data.get('business_name')
        token = data.get('bot_token')
        context = data.get('business_context')
        password = data.get('password')
        
        if not all([name, token, context, password]):
            return jsonify({"error": "All fields are required"}), 400
        if get_business_by_token(token):
            return jsonify({"error": "Token already registered"}), 409
        
        business_id = register_business(name, token, context, password)
        start_bot_thread(token, business_id)
        session['business_id'] = business_id
        session['business_name'] = name
        return jsonify({"status": "success", "business_id": business_id, "redirect": f"/dashboard/{business_id}"})
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/business/<int:business_id>')
def api_get_business(business_id):
    if 'business_id' not in session or session['business_id'] != business_id:
        return jsonify({"error": "Unauthorized"}), 401
    biz = get_business(business_id)
    if not biz:
        return jsonify({"error": "Not found"}), 404
    return jsonify(biz)

@app.route('/api/business/<int:business_id>', methods=['PUT'])
def api_update_business(business_id):
    if 'business_id' not in session or session['business_id'] != business_id:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    name = data.get('business_name')
    context = data.get('business_context')
    active = data.get('is_active')
    password = data.get('password')
    update_business(business_id, name=name, context=context, active=active, password=password)
    return jsonify({"status": "updated"})

@app.route('/api/messages/<int:business_id>')
def api_get_messages(business_id):
    if 'business_id' not in session or session['business_id'] != business_id:
        return jsonify({"error": "Unauthorized"}), 401
    limit = request.args.get('limit', 100, type=int)
    msgs = get_messages(business_id, limit)
    return jsonify(msgs)

@app.route('/api/analytics/<int:business_id>')
def api_get_analytics(business_id):
    if 'business_id' not in session or session['business_id'] != business_id:
        return jsonify({"error": "Unauthorized"}), 401
    stats = get_analytics(business_id)
    return jsonify(stats)

# ---------- DEBUG: Database initializer (access via browser) ----------
@app.route('/init-db')
def init_db_route():
    try:
        # Force table creation by calling get_db()
        with get_db() as conn:
            pass
        return "Database initialized successfully! ✅"
    except Exception as e:
        return f"Error: {e} ❌"

# ---------- STARTUP ----------
if __name__ == '__main__':
    # Ensure database is created on startup
    with get_db() as conn:
        pass  # tables are created automatically
    
    # Restart any active bots from the database
    with get_db() as conn:
        rows = conn.execute("SELECT id, bot_token FROM businesses WHERE is_active = 1").fetchall()
        for row in rows:
            start_bot_thread(row['bot_token'], row['id'])
    
    print("🌐 Web server running...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
