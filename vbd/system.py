"""The MVP service logic: enrolment, verification, liveness check, decision. Storage holds ONLY the voiceprint vector.

    auth = VoiceAuth(store_path="demo/templates.sqlite")
    auth.enroll("user-1", [wav1, wav2, wav3])           # raw audio is used in memory and never written anywhere
    auth.verify("user-1", wav)                           # -> dict(decision, sv_score, live_score, user_id, timestamp, ms)
"""
import os, sqlite3, time, datetime, numpy as np, torch
from . import config as C, sv as SV, cm as CM, augment as A


class TemplateStore:
    """SQLite table user_id -> 192-d float32 unit vector (768 bytes) + timestamps. No audio column exists."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS templates (user_id TEXT PRIMARY KEY, dim INTEGER, vec BLOB, n_utts INTEGER, created TEXT, updated TEXT)")
        self.db.commit()

    def put(self, user_id, vec, n_utts):
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        v = np.asarray(vec, np.float32)
        self.db.execute("INSERT INTO templates VALUES (?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET vec=excluded.vec, n_utts=excluded.n_utts, updated=excluded.updated",
                        (user_id, len(v), v.tobytes(), n_utts, now, now)); self.db.commit()

    def get(self, user_id):
        r = self.db.execute("SELECT vec FROM templates WHERE user_id=?", (user_id,)).fetchone()
        return None if r is None else np.frombuffer(r[0], np.float32)

    def users(self):
        return [r[0] for r in self.db.execute("SELECT user_id FROM templates ORDER BY user_id")]

    def delete(self, user_id):
        self.db.execute("DELETE FROM templates WHERE user_id=?", (user_id,)); self.db.commit()


class VoiceAuth:
    def __init__(self, store_path, sv_ckpt=f"{C.CKPT}/sv/stage2v2/best.pt", cm_model="lcnn", cm_ckpt=f"{C.CKPT}/cm/lcnn_asv19la/best.pt",
                 thr_sv=0.35, thr_live=0.0, device="cpu"):
        self.device = device
        self.enc = SV.load_encoder("ecapa", device=device)
        if sv_ckpt: self.enc.mods.embedding_model.load_state_dict(torch.load(sv_ckpt, map_location=device))
        self.enc.eval()
        self.cm = None
        if cm_ckpt and os.path.exists(cm_ckpt):
            self.cm = CM.build(cm_model).to(device); self.cm.load_state_dict(torch.load(cm_ckpt, map_location=device)); self.cm.eval()
        self.store = TemplateStore(store_path); self.thr_sv, self.thr_live = thr_sv, thr_live

    def _embed(self, wavs):
        return SV.embed(self.enc, [np.asarray(w, np.float32) for w in wavs])

    def enroll(self, user_id, wavs):
        v = self._embed(wavs).mean(0); v /= np.linalg.norm(v)
        self.store.put(user_id, v, len(wavs))
        return dict(user_id=user_id, n_utts=len(wavs), dim=len(v))

    def verify(self, user_id, wav):
        t0 = time.perf_counter()
        tpl = self.store.get(user_id)
        if tpl is None:
            return dict(decision="reject", reason="unknown user", user_id=user_id, sv_score=None, live_score=None,
                        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), ms=1000 * (time.perf_counter() - t0))
        wav = np.asarray(wav, np.float32)
        sv_score = float(self._embed([wav])[0] @ tpl)
        live = float(CM.score(self.cm, [wav], device=self.device)[0]) if self.cm is not None else None
        ok_sv = sv_score >= self.thr_sv; ok_live = True if live is None else live >= self.thr_live
        reason = "ok" if (ok_sv and ok_live) else ("speaker mismatch" if not ok_sv else "liveness check failed")
        return dict(decision="accept" if (ok_sv and ok_live) else "reject", reason=reason, user_id=user_id, sv_score=sv_score, live_score=live,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), ms=1000 * (time.perf_counter() - t0))

    def verify_file(self, user_id, path):
        """Same as verify() but timed from reading the audio file."""
        t0 = time.perf_counter(); out = self.verify(user_id, A.load(path)); out["ms"] = 1000 * (time.perf_counter() - t0)
        return out
