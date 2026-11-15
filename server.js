import express from 'express';
import zlib from 'zlib';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const PRIMARY_MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
const FALLBACK_MODEL = process.env.GEMINI_FALLBACK_MODEL || 'gemini-3.5-flash-lite';

const apiKey = process.env.GEMINI_API_KEY?.trim();
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

app.set('trust proxy', 1);
app.use(express.json({ limit: '256kb' }));

// ==========================================
// In-Memory PNG Icon Generator (for Android PWA)
// ==========================================
const crcTable = new Uint32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  crcTable[n] = c >>> 0;
}
function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function pngChunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const t = Buffer.from(type, 'ascii');
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([t, data])), 0);
  return Buffer.concat([len, t, data, crc]);
}
function makeIconPng(size) {
  const raw = Buffer.alloc((size * 4 + 1) * size);
  const cx = size / 2;
  const cy = size / 2;
  const outerR = size * 0.34;
  const innerR = size * 0.21;
  for (let y = 0; y < size; y++) {
    const row = y * (size * 4 + 1);
    raw[row] = 0;
    for (let x = 0; x < size; x++) {
      const i = row + 1 + x * 4;
      const dist = Math.hypot(x - cx, y - cy);
      const t = (x + y) / (2 * size);
      let r = Math.round(8 + t * 12);
      let g = Math.round(10 + t * 10);
      let b = Math.round(23 + t * 24);
      const aura = Math.max(0, 1 - dist / (size * 0.48));
      r = Math.min(255, r + Math.round(aura * aura * 55));
      g = Math.min(255, g + Math.round(aura * aura * 25));
      b = Math.min(255, b + Math.round(aura * aura * 95));
      if (dist <= outerR && dist >= innerR) {
        const gt = x / size;
        const fade = Math.min(1, Math.min(outerR - dist, dist - innerR) / (size * 0.025));
        r = Math.round(r * (1 - fade) + Math.round(139 * (1 - gt) + 59 * gt) * fade);
        g = Math.round(g * (1 - fade) + Math.round(92 * (1 - gt) + 130 * gt) * fade);
        b = Math.round(b * (1 - fade) + 246 * fade);
      }
      if (dist < size * 0.095) {
        const cf = Math.min(1, (size * 0.095 - dist) / (size * 0.02));
        r = Math.round(r * (1 - cf) + 224 * cf);
        g = Math.round(g * (1 - cf) + 231 * cf);
        b = Math.round(b * (1 - cf) + 255 * cf);
      }
      raw[i] = r; raw[i + 1] = g; raw[i + 2] = b; raw[i + 3] = 255;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0); ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', zlib.deflateSync(raw)),
    pngChunk('IEND', Buffer.alloc(0))
  ]);
}
const ICON_192 = makeIconPng(192);
const ICON_512 = makeIconPng(512);

// ==========================================
// Rate Limiter & Gemini System Prompt
// ==========================================
const buckets = new Map();
function rateLimit(req, res, next) {
  const ip = req.ip || 'local';
  const now = Date.now();
  const hits = (buckets.get(ip) || []).filter((t) => now - t < 60000);
  if (hits.length >= 20) {
    return res.status(429).json({ error: 'Too many messages right now. Please wait a few seconds.' });
  }
  hits.push(now);
  buckets.set(ip, hits);
  next();
}

function buildInstruction({ languageMode = 'auto', memoryEnabled = false, memories = [] }) {
  const langRule = {
    auto: 'Mirror the user language naturally: Tamil script if they type Tamil, Tanglish if they type Tanglish, or English if they type English.',
    tanglish: 'Reply mainly in warm, natural Tanglish (Tamil in English letters) with clear English technical terms.',
    tamil: 'Reply mainly in warm, natural Tamil script (தமிழ்).',
    english: 'Reply mainly in warm, clear English while still understanding Tamil and Tanglish.'
  }[languageMode] || 'Match the user language naturally.';

  const safeMems = memoryEnabled && Array.isArray(memories)
    ? memories.map((m) => String(m || '').trim().slice(0, 200)).filter(Boolean).slice(0, 20)
    : [];
  const memText = safeMems.length ? '\n\nUSER-APPROVED MEMORY:\n- ' + safeMems.join('\n- ') : '';

  return `You are PREETHI AI ("Preethi"), a warm, witty, supportive female personal AI companion.
- Friendly, natural, encouraging, and clear for mobile reading and browser Text-to-Speech.
- Fluent in Tamil (தமிழ்), Tanglish, and English.
- You natively understand Madurai-style colloquial Tamil ("ennanga", "epdi irukkeenga", "nallaa irukkaa", "vaanga", "sollunga", "aamaanga", "appadiyaa", "enna vishayam", "konjam", "theriyala", "seri vidu", "semma", "kalakkal", "thozhi", "pa/ma/le").
- ${langRule}
- Help with technical coding/debugging, educational concepts/study plans, and everyday advice.
- Privacy rule: never ask for passwords, OTPs, bank/card numbers, Aadhaar/PAN, or secrets.${memText}`;
}

// ==========================================
// Embedded PWA Frontend (Zero Extra Files)
// ==========================================
const ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><defs><linearGradient id="b" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#070814"/><stop offset="100%" stop-color="#131633"/></linearGradient><linearGradient id="r" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#a855f7"/><stop offset="100%" stop-color="#3b82f6"/></linearGradient></defs><rect width="512" height="512" rx="112" fill="url(#b)"/><circle cx="256" cy="256" r="136" fill="none" stroke="url(#r)" stroke-width="34"/><circle cx="256" cy="256" r="46" fill="#e0e7ff"/></svg>`;

const MANIFEST_JSON = JSON.stringify({
  name: 'PREETHI AI',
  short_name: 'Preethi AI',
  description: 'Your friendly Tamil, Tanglish & English personal AI companion.',
  start_url: '/',
  scope: '/',
  display: 'standalone',
  orientation: 'portrait',
  background_color: '#070814',
  theme_color: '#0b0d1f',
  icons: [
    { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any maskable' },
    { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' },
    { src: '/icons/icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' }
  ]
});

const SW_JS = `const C='preethi-v1';
self.addEventListener('install',e=>{e.waitUntil(caches.open(C).then(c=>c.addAll(['/','/manifest.webmanifest','/icons/icon-192.png','/icons/icon-512.png'])));self.skipWaiting();});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==C).map(k=>caches.delete(k)))));self.clients.claim();});
self.addEventListener('fetch',e=>{const u=new URL(e.request.url);if(u.pathname.startsWith('/api/')||e.request.method!=='GET')return;e.respondWith(caches.match(e.request).then(c=>c||fetch(e.request).catch(()=>c)));});`;

const CSS_TEXT = `
:root{--bg:#070814;--elev:rgba(18,21,44,.82);--card:rgba(24,28,58,.68);--bdr:rgba(148,163,255,.16);--tx:#f3f5ff;--tx2:#a5b0d6;--mut:#707ba6;--grad:linear-gradient(135deg,#8b5cf6,#3b82f6);--bub:linear-gradient(135deg,#7c3aed,#2563eb);--font:'Plus Jakarta Sans','Noto Sans Tamil',system-ui,sans-serif}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;height:100%;background:var(--bg);color:var(--tx);font-family:var(--font);overflow:hidden}
.hidden{display:none!important}
.bg-glow{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden}
.blob{position:absolute;border-radius:50%;filter:blur(90px);opacity:.28}
.bp{width:300px;height:300px;background:#8b5cf6;top:-80px;left:-80px}
.bb{width:320px;height:320px;background:#3b82f6;bottom:-90px;right:-80px}
.shell{position:relative;z-index:1;display:flex;flex-direction:column;height:100dvh;max-width:860px;margin:0 auto;padding-top:env(safe-area-inset-top)}
.top{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 12px;background:rgba(11,13,31,.88);backdrop-filter:blur(16px);border-bottom:1px solid var(--bdr)}
.row{display:flex;align-items:center;gap:8px}
.brand h1{margin:0;font-size:1rem;background:linear-gradient(90deg,#e9d5ff,#93c5fd);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{margin:2px 0 0;font-size:.72rem;color:var(--tx2)}
.orb{width:36px;height:36px;border-radius:50%;background:var(--grad);display:grid;place-items:center;box-shadow:0 0 18px rgba(139,92,246,.55);flex-shrink:0}
.orb.lg{width:64px;height:64px;margin:0 auto 12px}
.orb-c{width:42%;height:42%;border-radius:50%;background:#e0e7ff;box-shadow:0 0 10px #fff}
.orb.listen{animation:pulse 1.1s infinite}.orb.speak{animation:spk .9s infinite}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.12)}}
@keyframes spk{0%,100%{transform:scale(1)}50%{transform:scale(1.07);filter:brightness(1.2)}}
.ibtn{width:42px;height:42px;border:1px solid var(--bdr);border-radius:12px;background:rgba(255,255,255,.04);color:var(--tx);display:grid;place-items:center;cursor:pointer;font-size:1.2rem}
.ibtn svg{width:20px;height:20px}
.pbtn{padding:8px 13px;border-radius:999px;border:none;background:var(--grad);color:#fff;font-weight:600;font-size:.8rem;cursor:pointer}
.banner{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 14px;font-size:.82rem;background:rgba(244,63,94,.18);border-bottom:1px solid rgba(244,63,94,.35);color:#fecdd3}
.banner.info{background:rgba(59,130,246,.18);border-bottom-color:rgba(59,130,246,.35);color:#dbeafe}
.banner button{background:none;border:none;color:inherit;font-size:1.2rem;cursor:pointer}
.chat{flex:1;overflow-y:auto;padding:14px 12px 18px;overscroll-behavior-y:contain}
.welcome{margin:10px auto 16px;padding:20px 16px;max-width:560px;border-radius:20px;background:var(--card);border:1px solid var(--bdr);text-align:center}
.welcome h2{margin:0 0 8px;font-size:1.25rem}
.welcome p{margin:0 0 16px;font-size:.9rem;line-height:1.55;color:var(--tx2)}
.chips{display:grid;gap:8px}@media(min-width:560px){.chips{grid-template-columns:1fr 1fr}}
.chip{text-align:left;padding:11px 13px;border-radius:14px;border:1px solid var(--bdr);background:rgba(255,255,255,.04);color:var(--tx);font:inherit;font-size:.84rem;cursor:pointer}
.msgs{display:flex;flex-direction:column;gap:14px}
.m{display:flex;flex-direction:column;max-width:88%}@media(min-width:640px){.m{max-width:76%}}
.m.user{align-self:flex-end;align-items:flex-end}.m.model{align-self:flex-start;align-items:flex-start}
.bub{padding:12px 15px;border-radius:18px;font-size:.95rem;line-height:1.55;overflow-wrap:anywhere}
.m.user .bub{background:var(--bub);color:#fff;border-bottom-right-radius:5px}
.m.model .bub{background:var(--elev);border:1px solid var(--bdr);border-bottom-left-radius:5px}
.bub p{margin:0 0 8px}.bub p:last-child{margin-bottom:0}
.bub code{font-family:monospace;font-size:.85em;background:rgba(0,0,0,.35);padding:2px 6px;border-radius:6px}
.cb{margin:8px 0;border-radius:12px;background:#050711;border:1px solid var(--bdr);overflow:hidden}
.cbh{display:flex;justify-content:space-between;padding:6px 10px;font-size:.72rem;color:var(--tx2);background:rgba(255,255,255,.04)}
.cbh button{background:none;border:none;color:#93c5fd;cursor:pointer;font-size:.75rem}
.cb pre{margin:0;padding:10px 12px;overflow-x:auto;font-size:.82rem}
.meta{display:flex;align-items:center;gap:8px;margin-top:4px;font-size:.72rem;color:var(--mut)}
.meta button{background:none;border:none;color:var(--tx2);font-size:.74rem;padding:2px 6px;cursor:pointer}
.dots{display:inline-flex;gap:5px;padding:4px 2px}.dots span{width:7px;height:7px;border-radius:50%;background:#a5b4fc;animation:b 1.2s infinite}
.dots span:nth-child(2){animation-delay:.18s}.dots span:nth-child(3){animation-delay:.36s}
@keyframes b{0%,80%,100%{transform:scale(.6);opacity:.4}40%{transform:scale(1);opacity:1}}
.spkpill{align-self:center;display:flex;align-items:center;gap:10px;margin:4px 0 6px;padding:7px 14px;border-radius:999px;background:rgba(30,34,72,.94);border:1px solid rgba(139,92,246,.5);font-size:.8rem}
.spkpill button{border:none;background:rgba(244,63,94,.25);color:#fecdd3;padding:3px 10px;border-radius:999px;font-weight:600;cursor:pointer}
.comp{padding:8px 12px calc(10px + env(safe-area-inset-bottom));background:rgba(9,11,24,.94);border-top:1px solid var(--bdr)}
.tools{display:flex;gap:6px;overflow-x:auto;padding-bottom:8px;scrollbar-width:none}
.mpill{flex-shrink:0;padding:5px 10px;border-radius:999px;border:1px solid var(--bdr);background:rgba(255,255,255,.04);color:var(--tx2);font:inherit;font-size:.74rem;cursor:pointer}
.mpill.active{border-color:rgba(139,92,246,.6);background:rgba(139,92,246,.16);color:var(--tx)}
.bar{display:flex;align-items:flex-end;gap:8px;padding:6px 6px 6px 14px;border-radius:24px;background:rgba(24,28,58,.86);border:1px solid var(--bdr)}
#messageInput{flex:1;border:none;background:transparent;color:var(--tx);font:inherit;font-size:1rem;line-height:1.4;max-height:125px;padding:8px 0;resize:none;outline:none}
.abtn{width:44px;height:44px;border-radius:50%;border:none;display:grid;place-items:center;cursor:pointer;flex-shrink:0}
.abtn svg{width:20px;height:20px}
.mic{background:rgba(255,255,255,.08);color:var(--tx)}
.mic.listen{background:rgba(244,63,94,.26);color:#fda4af;box-shadow:0 0 0 5px rgba(244,63,94,.2)}
.mic.denied{color:#fb7185;border:1px solid rgba(244,63,94,.5)}
.send{background:var(--grad);color:#fff}.send:disabled{opacity:.4}
.spin{width:18px;height:18px;border:2px solid rgba(255,255,255,.35);border-top-color:#fff;border-radius:50%;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.ov{position:fixed;inset:0;z-index:40;background:rgba(3,4,12,.76);backdrop-filter:blur(8px);display:flex}
.drw{width:min(86vw,330px);height:100%;background:#0c0f24;border-right:1px solid var(--bdr);display:flex;flex-direction:column;padding:16px 14px;gap:12px}
.hlist{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px}
.hitem{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 12px;border-radius:12px;background:rgba(255,255,255,.04);cursor:pointer}
.hitem.active{border:1px solid rgba(139,92,246,.65);background:rgba(139,92,246,.14)}
.mov{align-items:flex-end;justify-content:center}@media(min-width:640px){.mov{align-items:center}}
.sheet{width:100%;max-width:510px;max-height:88dvh;overflow-y:auto;background:#0e1128;border:1px solid var(--bdr);border-radius:22px 22px 0 0;padding:18px 16px 24px;display:flex;flex-direction:column;gap:13px}
@media(min-width:640px){.sheet{border-radius:22px}}
.tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;background:rgba(255,255,255,.04);padding:4px;border-radius:12px}
.tbtn{border:none;background:transparent;color:var(--tx2);padding:9px 6px;border-radius:9px;font-weight:600;cursor:pointer}
.tbtn.active{background:var(--grad);color:#fff}
.tp{display:none;flex-direction:column;gap:12px}.tp.active{display:flex}
.fld{display:flex;flex-direction:column;gap:6px;font-size:.84rem;color:var(--tx2)}
.fld select,.madd input{width:100%;padding:11px;border-radius:12px;border:1px solid var(--bdr);background:rgba(255,255,255,.05);color:var(--tx);font:inherit}
.trow{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px;border-radius:14px;background:rgba(255,255,255,.04)}
.madd{display:flex;gap:8px}.mlist{display:flex;flex-direction:column;gap:8px;max-height:190px;overflow-y:auto}
.mitem{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-radius:12px;background:rgba(255,255,255,.05);font-size:.86rem}
.nbox{padding:11px;border-radius:12px;background:rgba(59,130,246,.14);border:1px solid rgba(59,130,246,.3);font-size:.8rem;color:#dbeafe}
.warn{margin:0;font-size:.8rem;color:#fda4af}
.pri{padding:12px 16px;border-radius:12px;border:none;background:var(--grad);color:#fff;font-weight:600;cursor:pointer;width:100%}
.sec{padding:12px 16px;border-radius:12px;border:1px solid var(--bdr);background:rgba(255,255,255,.08);color:var(--tx);font-weight:600;cursor:pointer;width:100%}
.dng{padding:12px 16px;border-radius:12px;border:1px solid rgba(244,63,94,.45);background:transparent;color:#fda4af;font-weight:600;cursor:pointer;width:100%}
`;

// Client JS uses zero backticks or template interpolation so embedding in server.js is 100% safe
const CLIENT_JS = `
var KEY = 'preethi_ai_state_v1';
var DEF = {
  privacyAccepted: false,
  activeChatId: null,
  chats: [],
  settings: { languageMode: 'auto', voiceLang: 'en-IN', autoSpeak: true, ttsVoiceURI: '', speechRate: 1.0, memoryEnabled: false },
  memories: []
};
var state = loadState();
var isSending = false, micState = 'idle', rec = null, deferredPrompt = null;

function $(id) { return document.getElementById(id); }
function loadState() {
  try {
    var r = localStorage.getItem(KEY);
    if (!r) return JSON.parse(JSON.stringify(DEF));
    var p = JSON.parse(r);
    return Object.assign({}, JSON.parse(JSON.stringify(DEF)), p, { settings: Object.assign({}, DEF.settings, p.settings || {}) });
  } catch (e) { return JSON.parse(JSON.stringify(DEF)); }
}
function save() { localStorage.setItem(KEY, JSON.stringify(state)); }
function activeChat() {
  var c = state.chats.find(function(x){ return x.id === state.activeChatId; });
  if (!c) {
    c = { id: String(Date.now()) + Math.random().toString(16).slice(2,6), title: 'New Chat', updatedAt: Date.now(), messages: [] };
    state.chats.unshift(c); state.activeChatId = c.id; save();
  }
  return c;
}
function hasSensitive(t) {
  return [/\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}(?:[\\s-]?\\d{4})?\\b/, /\\b[A-Z]{5}\\d{4}[A-Z]\\b/i, /\\b(password|passcode|otp|cvv|atm\\s*pin|secret\\s*key|api\\s*key|bank\\s*account)\\b/i].some(function(re){ return re.test(t); });
}
function banner(txt, info) {
  $('statusBannerText').textContent = txt;
  $('statusBanner').className = info ? 'banner info' : 'banner';
}
function hideBanner() { $('statusBanner').className = 'banner hidden'; }
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function formatReply(raw) {
  var blocks = [];
  var bt = String.fromCharCode(96);
  var fenceRe = new RegExp(bt + bt + bt + '(\\\\w*)\\\\n?([\\\\s\\\\S]*?)' + bt + bt + bt, 'g');
  var s = raw.replace(fenceRe, function(_m, lang, code) {
    var i = blocks.length; blocks.push({ lang: lang || 'code', code: code.trim() });
    return '%%CB_' + i + '%%';
  });
  s = esc(s).replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  s = s.replace(new RegExp(bt + '([^' + bt + ']+)' + bt, 'g'), '<code>$1</code>');
  s = s.split(/\\n{2,}/).map(function(p){ return '<p>' + p.replace(/\\n/g, '<br/>') + '</p>'; }).join('');
  return s.replace(/%%CB_(\\d+)%%/g, function(_m, idx) {
    var b = blocks[Number(idx)];
    return '<div class="cb"><div class="cbh"><span>' + esc(b.lang) + '</span><button type="button" class="copybtn" data-c="' + esc(b.code) + '">Copy</button></div><pre><code>' + esc(b.code) + '</code></pre></div>';
  });
}
function renderChat() {
  var c = activeChat();
  $('messagesList').innerHTML = '';
  $('welcomeState').classList.toggle('hidden', c.messages.length > 0);
  c.messages.forEach(addMsgEl);
  $('chatContainer').scrollTop = $('chatContainer').scrollHeight;
  renderHistory();
}
function addMsgEl(m) {
  var row = document.createElement('div');
  row.className = 'm ' + m.role;
  var bub = document.createElement('div');
  bub.className = 'bub';
  if (m.role === 'model') bub.innerHTML = formatReply(m.text);
  else bub.textContent = m.text;
  var meta = document.createElement('div');
  meta.className = 'meta';
  var tm = document.createElement('span');
  tm.textContent = new Date(m.ts || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  meta.appendChild(tm);
  if (m.role === 'model') {
    var sp = document.createElement('button'); sp.type = 'button'; sp.textContent = 'Speak';
    sp.onclick = function(){ speak(m.text); };
    var cp = document.createElement('button'); cp.type = 'button'; cp.textContent = 'Copy';
    cp.onclick = function(){ if (navigator.clipboard) navigator.clipboard.writeText(m.text); cp.textContent = 'Copied!'; setTimeout(function(){ cp.textContent='Copy'; }, 1000); };
    meta.append(sp, cp);
  }
  row.append(bub, meta);
  $('messagesList').appendChild(row);
}
function send(override) {
  if (isSending) return;
  var txt = (override != null ? override : $('messageInput').value).trim();
  if (!txt) return;
  if (!navigator.onLine) { banner('You are offline. Connect to internet to chat.'); return; }
  hideBanner(); stopSpeak();
  var c = activeChat();
  var u = { role: 'user', text: txt, ts: Date.now() };
  c.messages.push(u);
  if (c.title === 'New Chat') c.title = txt.slice(0, 34) + (txt.length > 34 ? '...' : '');
  c.updatedAt = Date.now(); save();
  $('welcomeState').classList.add('hidden');
  addMsgEl(u);
  if (override == null) { $('messageInput').value = ''; resizeBox(); }
  updateSend();
  isSending = true; setLoad(true);
  var tr = document.createElement('div'); tr.id = 'typingRow'; tr.className = 'm model';
  tr.innerHTML = '<div class="bub"><div class="dots"><span></span><span></span><span></span></div></div>';
  $('messagesList').appendChild(tr);
  $('chatContainer').scrollTop = $('chatContainer').scrollHeight;

  fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: c.messages.slice(-20).map(function(x){ return { role: x.role, text: x.text }; }),
      languageMode: state.settings.languageMode,
      memoryEnabled: state.settings.memoryEnabled,
      memories: state.settings.memoryEnabled ? state.memories : []
    })
  }).then(function(res){ return res.json().then(function(d){ if(!res.ok) throw new Error(d.error || 'Server error'); return d; }); })
    .then(function(d){
      var el = $('typingRow'); if (el) el.remove();
      var mm = { role: 'model', text: d.reply, ts: Date.now() };
      c.messages.push(mm); c.updatedAt = Date.now(); save();
      addMsgEl(mm); renderHistory();
      $('chatContainer').scrollTop = $('chatContainer').scrollHeight;
      if (state.settings.autoSpeak) speak(d.reply);
    }).catch(function(err){
      var el = $('typingRow'); if (el) el.remove();
      banner(err.message || 'Could not reach Preethi AI.');
    }).finally(function(){ isSending = false; setLoad(false); updateSend(); });
}
function setLoad(on) {
  $('sendIcon').classList.toggle('hidden', on);
  $('sendSpin').classList.toggle('hidden', !on);
  $('statusSubtitle').textContent = on ? 'Preethi is thinking...' : 'Ready • Tamil · Tanglish · English';
}
function updateSend() { $('sendBtn').disabled = isSending || !$('messageInput').value.trim(); }
function resizeBox() { $('messageInput').style.height = 'auto'; $('messageInput').style.height = Math.min($('messageInput').scrollHeight, 125) + 'px'; }

function initMic() {
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { micState = 'unsupported'; return; }
  rec = new SR(); rec.continuous = false; rec.interimResults = true;
  rec.onstart = function(){ micState = 'listening'; $('micBtn').className = 'abtn mic listen'; $('preethiOrb').classList.add('listen'); $('statusSubtitle').textContent = 'Listening (' + state.settings.voiceLang + ')...'; };
  rec.onresult = function(e){ var t = ''; for(var i=0;i<e.results.length;i++) t += e.results[i][0].transcript; $('messageInput').value = t; resizeBox(); updateSend(); };
  rec.onerror = function(e){
    if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
      micState = 'denied'; $('micBtn').className = 'abtn mic denied';
      banner('Microphone permission denied. Tap the lock icon in Chrome address bar -> Allow Microphone.');
    } else if (e.error === 'no-speech') banner('No speech detected. Tap mic and try again.', true);
    else if (e.error !== 'aborted') banner('Voice error: ' + e.error);
  };
  rec.onend = function(){ if (micState === 'listening') micState = 'idle'; $('micBtn').className = 'abtn mic'; $('preethiOrb').classList.remove('listen'); $('statusSubtitle').textContent = 'Ready • Tamil · Tanglish · English'; };
}
function toggleMic() {
  if (micState === 'unsupported' || !rec) { banner('Voice input requires Android Chrome.'); return; }
  if (micState === 'listening') { rec.stop(); return; }
  stopSpeak(); hideBanner(); micState = 'requesting';
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(function(st){
      st.getTracks().forEach(function(t){ t.stop(); });
      rec.lang = state.settings.voiceLang; rec.start();
    }).catch(function(){
      micState = 'denied'; $('micBtn').className = 'abtn mic denied';
      banner('Microphone blocked. Allow Microphone in Android Chrome Site Settings.');
    });
  } else { rec.lang = state.settings.voiceLang; rec.start(); }
}

function loadVoices() {
  if (!('speechSynthesis' in window)) return;
  var vs = speechSynthesis.getVoices();
  var sel = $('ttsVoiceSelect'); sel.innerHTML = '<option value="">Auto (Tamil for தமிழ், EN-IN otherwise)</option>';
  vs.forEach(function(v){
    var o = document.createElement('option'); o.value = v.voiceURI; o.textContent = v.name + ' (' + v.lang + ')';
    if (v.voiceURI === state.settings.ttsVoiceURI) o.selected = true;
    sel.appendChild(o);
  });
}
function pickVoice(txt) {
  var vs = speechSynthesis.getVoices(); if (!vs.length) return null;
  if (/[\\u0B80-\\u0BFF]/.test(txt)) { var tv = vs.find(function(v){ return v.lang.toLowerCase().indexOf('ta') === 0; }); if (tv) return tv; }
  if (state.settings.ttsVoiceURI) { var cv = vs.find(function(v){ return v.voiceURI === state.settings.ttsVoiceURI; }); if (cv) return cv; }
  return vs.find(function(v){ return v.lang.toLowerCase() === 'en-in'; }) || vs[0];
}
function speak(raw) {
  if (!('speechSynthesis' in window)) return;
  stopSpeak();
  var bt = String.fromCharCode(96);
  var clean = raw.replace(new RegExp(bt+bt+bt+'[\\\\s\\\\S]*?'+bt+bt+bt, 'g'), ' Code shown on screen. ').replace(/[*_#>-]/g, ' ').replace(/\\s+/g, ' ').trim();
  if (!clean) return;
  var parts = clean.match(/[^.!?\\n]+[.!?\\n]?/g) || [clean];
  var voice = pickVoice(clean);
  $('speakingPill').classList.remove('hidden'); $('preethiOrb').classList.add('speak');
  parts = parts.map(function(p){ return p.trim(); }).filter(Boolean);
  parts.forEach(function(p, idx){
    var u = new SpeechSynthesisUtterance(p);
    if (voice) { u.voice = voice; u.lang = voice.lang; }
    u.rate = Number(state.settings.speechRate) || 1;
    if (idx === parts.length - 1) {
      u.onend = function(){ $('speakingPill').classList.add('hidden'); $('preethiOrb').classList.remove('speak'); };
      u.onerror = u.onend;
    }
    speechSynthesis.speak(u);
  });
}
function stopSpeak() {
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  $('speakingPill').classList.add('hidden'); $('preethiOrb').classList.remove('speak');
}

function renderHistory() {
  var box = $('historyList'); box.innerHTML = '';
  state.chats.forEach(function(c){
    var d = document.createElement('div'); d.className = 'hitem' + (c.id === state.activeChatId ? ' active' : '');
    var info = document.createElement('div'); info.style.minWidth = '0';
    info.innerHTML = '<div style="font-size:.88rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + esc(c.title) + '</div><div style="font-size:.72rem;color:#707ba6">' + new Date(c.updatedAt).toLocaleDateString() + ' • ' + c.messages.length + ' msgs</div>';
    info.onclick = function(){ state.activeChatId = c.id; save(); renderChat(); $('historyOverlay').classList.add('hidden'); };
    var del = document.createElement('button'); del.textContent = '×'; del.style.cssText = 'background:none;border:none;color:#707ba6;font-size:1.2rem;cursor:pointer';
    del.onclick = function(e){ e.stopPropagation(); state.chats = state.chats.filter(function(x){ return x.id !== c.id; }); if(state.activeChatId===c.id) state.activeChatId = state.chats[0]?state.chats[0].id:null; save(); renderChat(); };
    d.append(info, del); box.appendChild(d);
  });
}
function renderMemory() {
  var box = $('memoryList'); box.innerHTML = '';
  if (!state.memories.length) { box.innerHTML = '<p style="font-size:.78rem;color:#707ba6;margin:0">No memories saved yet.</p>'; return; }
  state.memories.forEach(function(m, i){
    var d = document.createElement('div'); d.className = 'mitem';
    var sp = document.createElement('span'); sp.textContent = m;
    var del = document.createElement('button'); del.textContent = '×'; del.style.cssText = 'background:none;border:none;color:#a5b0d6;font-size:1.1rem;cursor:pointer';
    del.onclick = function(){ state.memories.splice(i, 1); save(); renderMemory(); };
    d.append(sp, del); box.appendChild(d);
  });
}
function syncUI() {
  $('languageModeSelect').value = state.settings.languageMode;
  $('voiceLangSelect').value = state.settings.voiceLang;
  $('autoSpeakToggle').checked = state.settings.autoSpeak;
  $('speechRateRange').value = state.settings.speechRate;
  $('speechRateVal').textContent = Number(state.settings.speechRate).toFixed(2) + 'x';
  $('memoryConsentToggle').checked = state.settings.memoryEnabled;
  $('voiceLangLabel').textContent = state.settings.voiceLang === 'ta-IN' ? 'TA-IN' : 'EN-IN';
  $('autoSpeakLabel').textContent = state.settings.autoSpeak ? 'ON' : 'OFF';
  $('autoSpeakChip').classList.toggle('active', state.settings.autoSpeak);
  renderMemory();
}
function openTab(t) {
  document.querySelectorAll('.tbtn').forEach(function(b){ b.classList.toggle('active', b.dataset.tab === t); });
  document.querySelectorAll('.tp').forEach(function(p){ p.classList.toggle('active', p.id === 'tab-' + t); });
  syncUI(); $('settingsModal').classList.remove('hidden');
}

$('chatForm').addEventListener('submit', function(e){ e.preventDefault(); send(); });
$('messageInput').addEventListener('input', function(){ resizeBox(); updateSend(); });
document.querySelectorAll('.chip').forEach(function(c){ c.onclick = function(){ send(c.dataset.prompt); }; });
$('messagesList').addEventListener('click', function(e){ var b = e.target.closest('.copybtn'); if(b && navigator.clipboard){ navigator.clipboard.writeText(b.dataset.c); b.textContent='Copied!'; setTimeout(function(){b.textContent='Copy';},1000); } });
$('micBtn').onclick = toggleMic;
$('stopSpeakBtn').onclick = stopSpeak;
$('statusBannerClose').onclick = hideBanner;
$('voiceLangChip').onclick = function(){ state.settings.voiceLang = state.settings.voiceLang === 'en-IN' ? 'ta-IN' : 'en-IN'; save(); syncUI(); };
$('autoSpeakChip').onclick = function(){ state.settings.autoSpeak = !state.settings.autoSpeak; if(!state.settings.autoSpeak) stopSpeak(); save(); syncUI(); };
$('clearChatBtn').onclick = function(){ var c = activeChat(); if(c.messages.length && confirm('Clear this chat?')){ stopSpeak(); c.messages=[]; c.title='New Chat'; save(); renderChat(); } };
$('historyBtn').onclick = function(){ renderHistory(); $('historyOverlay').classList.remove('hidden'); };
$('closeHistoryBtn').onclick = function(){ $('historyOverlay').classList.add('hidden'); };
$('newChatBtn').onclick = function(){ var c = { id: String(Date.now()), title: 'New Chat', updatedAt: Date.now(), messages: [] }; state.chats.unshift(c); state.activeChatId = c.id; save(); renderChat(); $('historyOverlay').classList.add('hidden'); };
$('clearAllHistoryBtn').onclick = function(){ if(confirm('Delete ALL chats?')){ state.chats=[]; state.activeChatId=null; save(); renderChat(); $('historyOverlay').classList.add('hidden'); } };
$('settingsBtn').onclick = function(){ openTab('prefs'); };
$('privacyQuickBtn').onclick = function(){ openTab('privacy'); };
$('closeSettingsBtn').onclick = function(){ $('settingsModal').classList.add('hidden'); };
document.querySelectorAll('.tbtn').forEach(function(b){ b.onclick = function(){ openTab(b.dataset.tab); }; });
$('languageModeSelect').onchange = function(e){ state.settings.languageMode = e.target.value; save(); };
$('voiceLangSelect').onchange = function(e){ state.settings.voiceLang = e.target.value; save(); syncUI(); };
$('autoSpeakToggle').onchange = function(e){ state.settings.autoSpeak = e.target.checked; if(!e.target.checked) stopSpeak(); save(); syncUI(); };
$('ttsVoiceSelect').onchange = function(e){ state.settings.ttsVoiceURI = e.target.value; save(); };
$('speechRateRange').oninput = function(e){ state.settings.speechRate = Number(e.target.value); $('speechRateVal').textContent = state.settings.speechRate.toFixed(2)+'x'; save(); };
$('memoryConsentToggle').onchange = function(e){ state.settings.memoryEnabled = e.target.checked; save(); };
$('addMemoryBtn').onclick = function(){
  var v = $('memoryInput').value.trim(); $('memoryWarning').classList.add('hidden'); if(!v) return;
  if(hasSensitive(v)){ $('memoryWarning').textContent='Blocked: Looks like sensitive data (password/OTP/ID/card).'; $('memoryWarning').classList.remove('hidden'); return; }
  state.settings.memoryEnabled = true; state.memories.push(v); $('memoryInput').value=''; save(); syncUI();
};
$('wipeMemoryBtn').onclick = function(){ if(confirm('Clear all memory?')){ state.memories=[]; save(); renderMemory(); } };
$('exportDataBtn').onclick = function(){ var a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify(state.chats,null,2)],{type:'application/json'})); a.download='preethi-chats.json'; a.click(); };
$('resetAllDataBtn').onclick = function(){ if(confirm('Erase all local data?')){ stopSpeak(); localStorage.removeItem(KEY); state=loadState(); state.privacyAccepted=true; save(); syncUI(); renderChat(); $('settingsModal').classList.add('hidden'); } };
$('acceptPrivacyBtn').onclick = function(){ state.privacyAccepted=true; state.settings.memoryEnabled=$('initialMemoryConsent').checked; save(); syncUI(); $('privacyModal').classList.add('hidden'); };

window.addEventListener('beforeinstallprompt', function(e){ e.preventDefault(); deferredPrompt = e; $('installBtn').classList.remove('hidden'); });
$('installBtn').onclick = function(){ if(!deferredPrompt) return; deferredPrompt.prompt(); deferredPrompt = null; $('installBtn').classList.add('hidden'); };
window.addEventListener('offline', function(){ banner('Offline: you can view history, but chatting needs internet.'); });
window.addEventListener('online', hideBanner);

initMic(); loadVoices();
if ('speechSynthesis' in window) speechSynthesis.onvoiceschanged = loadVoices;
syncUI(); renderChat();
if (!state.privacyAccepted) $('privacyModal').classList.remove('hidden');
if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(function(){});
fetch('/api/health').then(function(r){ return r.json(); }).then(function(h){ if(!h.configured) banner('Server is missing GEMINI_API_KEY in environment settings.'); }).catch(function(){});
`;

const HTML_PAGE = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,interactive-widget=resizes-content"/>
<meta name="theme-color" content="#0b0d1f"/>
<meta name="mobile-web-app-capable" content="yes"/>
<title>PREETHI AI</title>
<link rel="manifest" href="/manifest.webmanifest"/>
<link rel="icon" href="/icons/icon-192.png"/>
<link rel="apple-touch-icon" href="/icons/icon-192.png"/>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil:wght@400;600&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>${CSS_TEXT}</style>
</head>
<body>
<div class="bg-glow"><div class="blob bp"></div><div class="blob bb"></div></div>
<div class="shell">
  <header class="top">
    <div class="row">
      <button id="historyBtn" class="ibtn" aria-label="History"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h12M4 18h16" stroke-linecap="round"/></svg></button>
      <div class="row">
        <div id="preethiOrb" class="orb"><span class="orb-c"></span></div>
        <div class="brand"><h1>PREETHI AI</h1><p id="statusSubtitle" class="sub">Ready • Tamil · Tanglish · English</p></div>
      </div>
    </div>
    <div class="row">
      <button id="installBtn" class="pbtn hidden">Install</button>
      <button id="clearChatBtn" class="ibtn" aria-label="Clear chat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4h8v2m-9 0v14a2 2 0 002 2h6a2 2 0 002-2V6" stroke-linecap="round"/></svg></button>
      <button id="settingsBtn" class="ibtn" aria-label="Settings">⚙️</button>
    </div>
  </header>

  <div id="statusBanner" class="banner hidden"><span id="statusBannerText"></span><button id="statusBannerClose">×</button></div>

  <main id="chatContainer" class="chat">
    <section id="welcomeState" class="welcome">
      <div class="orb lg"><span class="orb-c"></span></div>
      <h2>Vanakkam! I’m Preethi ✨</h2>
      <p>Ask me anything in <strong>English</strong>, <strong>Tanglish</strong>, or <strong>Madurai-style தமிழ்</strong>—coding, study help, or everyday life!</p>
      <div class="chips">
        <button class="chip" data-prompt="Ennanga Preethi, epdi irukkeenga? Innaiku semma fresh-aa oru productivity tip sollunga!">💜 Madurai Tanglish chat</button>
        <button class="chip" data-prompt="எனக்கு JavaScript async/await-ஐ எளிமையா உதாரணத்தோட சொல்லிக் கொடுங்க.">💻 தமிழ்ல Coding கத்துக்கலாம்</button>
        <button class="chip" data-prompt="Help me make a smart 2-hour evening study plan.">📚 Study plan helper</button>
        <button class="chip" data-prompt="Madurai style-la veetla easy-aa enna dinner seiyalam?">🍛 Everyday food idea</button>
      </div>
    </section>
    <div id="messagesList" class="msgs"></div>
  </main>

  <div id="speakingPill" class="spkpill hidden"><span>🔊 Preethi is speaking...</span><button id="stopSpeakBtn">Stop</button></div>

  <footer class="comp">
    <div class="tools">
      <button id="voiceLangChip" class="mpill" type="button">🎙️ Mic: <strong id="voiceLangLabel">EN-IN</strong></button>
      <button id="autoSpeakChip" class="mpill active" type="button">🔊 Auto-Speak: <strong id="autoSpeakLabel">ON</strong></button>
      <button id="privacyQuickBtn" class="mpill" type="button">🛡️ Privacy</button>
    </div>
    <form id="chatForm" class="bar">
      <textarea id="messageInput" rows="1" placeholder="Message Preethi (English, Tanglish, தமிழ்)..." enterkeyhint="send"></textarea>
      <button id="micBtn" type="button" class="abtn mic" aria-label="Voice input"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10a7 7 0 0014 0M12 17v5M8 22h8" stroke-linecap="round"/></svg></button>
      <button id="sendBtn" type="submit" class="abtn send" disabled aria-label="Send"><svg id="sendIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" stroke-linecap="round" stroke-linejoin="round"/></svg><span id="sendSpin" class="spin hidden"></span></button>
    </form>
  </footer>
</div>

<div id="historyOverlay" class="ov hidden">
  <aside class="drw">
    <div style="display:flex;justify-content:space-between;align-items:center"><h2 style="margin:0;font-size:1.1rem">Conversations</h2><button id="closeHistoryBtn" class="ibtn">×</button></div>
    <button id="newChatBtn" class="pri">+ New Chat</button>
    <div id="historyList" class="hlist"></div>
    <button id="clearAllHistoryBtn" class="dng">Delete All Chats</button>
  </aside>
</div>

<div id="settingsModal" class="ov mov hidden">
  <div class="sheet">
    <div style="display:flex;justify-content:space-between;align-items:center"><h2 style="margin:0;font-size:1.1rem">Settings &amp; Memory</h2><button id="closeSettingsBtn" class="ibtn">×</button></div>
    <div class="tabs">
      <button class="tbtn active" data-tab="prefs">Voice</button>
      <button class="tbtn" data-tab="memory">Memory</button>
      <button class="tbtn" data-tab="privacy">Privacy</button>
    </div>
    <div id="tab-prefs" class="tp active">
      <div class="fld"><label>Reply Language</label><select id="languageModeSelect"><option value="auto">Auto (Tamil / Tanglish / English)</option><option value="tanglish">Prefer Tanglish</option><option value="tamil">Prefer Tamil (தமிழ்)</option><option value="english">Prefer English</option></select></div>
      <div class="fld"><label>Mic Language</label><select id="voiceLangSelect"><option value="en-IN">English (India / Tanglish)</option><option value="ta-IN">Tamil (தமிழ்)</option></select></div>
      <div class="trow"><div><strong>Auto-Speak Replies</strong></div><input type="checkbox" id="autoSpeakToggle" checked/></div>
      <div class="fld"><label>TTS Voice</label><select id="ttsVoiceSelect"></select></div>
      <div class="fld"><label>Speed: <span id="speechRateVal">1.00x</span></label><input type="range" id="speechRateRange" min="0.75" max="1.35" step="0.05" value="1.0"/></div>
    </div>
    <div id="tab-memory" class="tp">
      <div class="nbox">Stored locally in your browser. Never save passwords, OTPs, bank numbers, or Aadhaar.</div>
      <div class="trow"><div><strong>Enable Memory (Consent)</strong></div><input type="checkbox" id="memoryConsentToggle"/></div>
      <div class="madd"><input id="memoryInput" placeholder="e.g., My name is Karthik" maxlength="200"/><button id="addMemoryBtn" class="pri" style="width:auto">Save</button></div>
      <p id="memoryWarning" class="warn hidden"></p>
      <div id="memoryList" class="mlist"></div>
      <button id="wipeMemoryBtn" class="dng">Clear All Memory</button>
    </div>
    <div id="tab-privacy" class="tp">
      <div class="nbox">Chats stay on your device. Prompts go securely to our backend using the Gemini Free Tier (API key is hidden on server). Never share sensitive secrets.</div>
      <button id="exportDataBtn" class="sec">Export Chats (JSON)</button>
      <button id="resetAllDataBtn" class="dng">Erase All Local Data</button>
    </div>
  </div>
</div>

<div id="privacyModal" class="ov mov hidden">
  <div class="sheet">
    <h2 style="margin:0">Welcome to PREETHI AI 🛡️</h2>
    <p style="margin:0;font-size:.88rem;color:#a5b0d6;line-height:1.5">Your chats stay on this device, and responses use the Gemini API Free Tier securely via our server. Please never share passwords, OTPs, Aadhaar, or bank details.</p>
    <label style="display:flex;gap:10px;font-size:.85rem"><input type="checkbox" id="initialMemoryConsent"/><span>Optional: Enable user-controlled Memory (OFF by default)</span></label>
    <button id="acceptPrivacyBtn" class="pri">I Understand — Start Chatting</button>
  </div>
</div>
<script>${CLIENT_JS}</script>
</body>
</html>`;

// ==========================================
// Routes
// ==========================================
app.get('/', (_req, res) => res.type('html').send(HTML_PAGE));
app.get('/manifest.webmanifest', (_req, res) => res.type('application/manifest+json').send(MANIFEST_JSON));
app.get('/sw.js', (_req, res) => res.set('Cache-Control', 'no-cache').type('application/javascript').send(SW_JS));
app.get('/icons/icon.svg', (_req, res) => res.type('image/svg+xml').send(ICON_SVG));
app.get('/icons/icon-192.png', (_req, res) => res.type('image/png').send(ICON_192));
app.get('/icons/icon-512.png', (_req, res) => res.type('image/png').send(ICON_512));

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, configured: Boolean(ai), model: PRIMARY_MODEL });
});

app.post('/api/chat', rateLimit, async (req, res) => {
  if (!ai) {
    return res.status(503).json({ error: 'Server missing GEMINI_API_KEY environment variable.' });
  }
  try {
    const { messages, languageMode, memoryEnabled, memories } = req.body || {};
    if (!Array.isArray(messages) || !messages.length) {
      return res.status(400).json({ error: 'Messages array is required.' });
    }
    const contents = messages
      .slice(-20)
      .map((m) => ({
        role: m.role === 'model' ? 'model' : 'user',
        parts: [{ text: String(m.text || '').trim().slice(0, 4000) }]
      }))
      .filter((m) => m.parts[0].text.length > 0);

    if (!contents.length || contents[contents.length - 1].role !== 'user') {
      return res.status(400).json({ error: 'Last message must be user text.' });
    }

    const config = {
      systemInstruction: buildInstruction({ languageMode, memoryEnabled: Boolean(memoryEnabled), memories }),
      temperature: 0.75,
      maxOutputTokens: 1200
    };

    let response;
    try {
      response = await ai.models.generateContent({ model: PRIMARY_MODEL, contents, config });
    } catch (err) {
      if ((err?.status === 429 || err?.status === 503) && FALLBACK_MODEL) {
        response = await ai.models.generateContent({ model: FALLBACK_MODEL, contents, config });
      } else {
        throw err;
      }
    }

    const reply = response?.text?.trim();
    if (!reply) return res.status(502).json({ error: 'Empty response from Gemini. Please try again.' });
    res.json({ reply });
  } catch (err) {
    console.error('Gemini error:', err?.status || '', err?.message || err);
    if (Number(err?.status) === 429) {
      return res.status(429).json({ error: 'Free-tier rate limit reached. Wait 30 seconds and try again.' });
    }
    res.status(500).json({ error: 'Error talking to Gemini. Check your API key or try again.' });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`PREETHI AI running on port ${PORT}`);
});