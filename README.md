# 🤖 AI Trading Bot

Sun'iy intellekt asosidagi avtomatik treyder - Exness MT5 uchun.  
XGBoost, LSTM, RL Agent va Technical Analysis modellarini birlashtirgan ensemble tizim.

---

## 🚀 Imkoniyatlar

### 📊 AI Analiz
- **XGBoost** - Gradient boosting bilan narx harakatini bashorat qilish
- **LSTM** - Uzoq muddatli xotiraga ega neyron tarmoq
- **RL Agent** - Reinforcement Learning orqali optimallashtirish
- **Technical Analysis** - 50+ indikator (RSI, MACD, Bollinger, ATR, ADX va b.)
- **Ensemble** - Barcha modellarni vaznli ovoz berish orqali birlashtirish

### 🎯 Smart Position Manager (YANGI!)
Ochiq pozitsiyalarni aqlli boshqarish tizimi:
- **Breakeven SL** - 15 pip foydada SL ni kirish narxiga suradi
- **Trailing Stop** - 25 pip foydada SL ni sudrab boradi
- **Partial Profit** - 30/50 pipda qisman foydani yopadi (30%+30%)
- **Profit Target** - 40 pip foydada butun pozitsiyani yopadi
- **Reversal Detection** - Bozor teskari ketayotganini sezsa pozitsiyani yopadi
- **AI Reversal** - AI model qarama-qarshi signal bersa yopadi
- **ATR Adjust** - Volatilikka qarab SL ni moslashtiradi

### 📡 Telegram Monitoring
- `/status` - Bot holati
- `/balance` - Account balansi
- `/positions` - Ochiq pozitsiyalar
- `/risk` - Risk hisoboti
- `/smart` - Smart Position Manager statusi
- `/pause` - Savdoni to'xtatish
- `/resume` - Savdoni davom ettirish
- `/mode` - Rejim almashtirish (normal/scalping)
- `/train` - AI modellarni qayta o'rgatish
- `/help` - Barcha buyruqlar

### ⚡ Scalping Mode
- M5 timeframe da tez savdolar
- Kichik SL/TP (ATR*0.5 / ATR*1.0)
- Minimal pozitsiya hajmi

---

## 🔧 O'rnatish

### 1. Talablar
- Python 3.10+
- MetaTrader 5 terminali (Exness hisobi bilan)
- Telegram bot tokeni (@BotFather orqali)

### 2. Loyihani yuklab olish
```bash
git clone https://github.com/YOUR_USERNAME/ai-trading-bot.git
cd ai-trading-bot
```

### 3. Virtual muhit yaratish
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 4. Kutubxonalarni o'rnatish
```bash
pip install -r requirements.txt
```

### 5. Sozlamalarni kiritish
```bash
cp .env.example .env
# .env faylini ochib o'z ma'lumotlaringizni kiriting:
#   - EXNESS_ACCOUNT, EXNESS_PASSWORD, EXNESS_SERVER
#   - TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

---

## 🏃 Ishga tushirish

```bash
python main.py
```

Bot avtomatik ravishda:
1. Exness MT5 ga ulanadi
2. AI modellarni yuklaydi
3. Ma'lumotlarni yig'ishni boshlaydi
4. Telegram botini ishga tushiradi
5. Trading siklini boshlaydi

---

## 📁 Loyiha tuzilishi

```
ai-trading-bot/
├── main.py                 # Asosiy ishga tushirish fayli
├── config/
│   ├── settings.py         # Global sozlamalar
│   └── .env.example       # Sozlamalar namunasi
├── src/
│   ├── trading_bot.py      # Asosiy bot (orchestrator)
│   ├── brokers/
│   │   ├── base_broker.py  # Broker interfeysi
│   │   └── mt5_broker.py   # MT5 broker integratsiyasi
│   ├── strategies/
│   │   ├── ensemble.py     # Ensemble qaror tizimi
│   │   ├── technical_analysis.py  # 50+ indikator
│   │   └── scalping_strategy.py   # Skalping strategiyasi
│   ├── ml/
│   │   ├── xgboost_model.py
│   │   └── lstm_model.py
│   ├── rl/
│   │   ├── rl_agent.py
│   │   └── trading_env.py
│   ├── risk/
│   │   ├── risk_manager.py          # Risk boshqaruvi
│   │   └── smart_position_manager.py  # Aqlli pozitsiya boshqaruvi 🆕
│   ├── data/
│   │   └── data_collector.py
│   └── monitoring/
│       └── telegram_bot.py  # Telegram monitoring
├── data/
│   ├── historical/          # Tarixiy ma'lumotlar
│   └── models/              # Saqlangan AI modellar
├── logs/                    # Log fayllar
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Sozlash

### Trading parametrlari
`.env` faylida o'zgartirish mumkin:

| Parametr | Standart | Tavsif |
|----------|----------|--------|
| `SYMBOLS` | XAUUSDm | Savdo qilinadigan symbol (vergul bilan ajrating) |
| `TIMEFRAMES` | M5,M15,H1,H4,D1 | Tahlil qilinadigan timeframe lar |
| `MAX_OPEN_TRADES` | 3 | Maksimal ochiq pozitsiyalar soni |
| `RISK_PER_TRADE` | 2.0 | Har bir savdo uchun risk (balansga nisbatan %) |
| `STOP_LOSS_ATR_MULTIPLIER` | 1.5 | Stop Loss ATR koeffitsiyenti |
| `TAKE_PROFIT_ATR_MULTIPLIER` | 3.0 | Take Profit ATR koeffitsiyenti |

### Smart Position Manager sozlamalari
`src/risk/smart_position_manager.py` da:

| Parametr | Standart | Tavsif |
|----------|----------|--------|
| `be_profit_pips` | 15 | Breakeven uchun minimal pip foyda |
| `trail_activation_pips` | 25 | Trailing aktivlashuvi uchun pip |
| `trail_distance_pips` | 15 | Trailing SL masofasi (pips) |
| `partial_profit_pips` | 30 | Birinchi qisman yopish darajasi |
| `partial_close_ratio` | 0.3 | Birinchi qisman yopish hajmi (30%) |
| `profit_target_pips` | 40 | To'liq yopish darajasi |
| `reversal_check_enabled` | True | Reversal tekshiruvi yoqilgan |

---

## 🤝 Hissa qo'shish

1. Fork qiling
2. Feature branch yarating (`git checkout -b feature/new-feature`)
3. O'zgartirishlarni commit qiling (`git commit -m 'Add new feature'`)
4. Push qiling (`git push origin feature/new-feature`)
5. Pull Request oching

---

## ⚠️ Eslatma

Bu bot **real pul** bilan ishlaydi. Ishlatishdan oldin:
1. **Demo hisobda sinab ko'ring** (Exness-MT5Trial server)
2. Kichik risk bilan boshlang (`RISK_PER_TRADE=0.5`)
3. Botni qarovsiz qoldirmang
4. Doimiy monitoring qilib turing

**Mas'uliyatni rad etish:** Muallif har qanday moliyaviy yo'qotishlar uchun javobgar emas. Botdan foydalanish butunlay o'z riskingizda.

---

## 📞 Aloqa

- Telegram: [@Muhammadamin_Ozadov](https://t.me/Muhammadamin_Ozadov)

---

## 📄 Litsenziya

MIT License - erkin foydalaning, o'zgartiring va tarqating.
