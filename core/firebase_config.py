"""
Firebase & Google Analytics Integration for SmartResearch Studio
Configured for project: ung-dung-da-nckh
Measurement ID: G-17SYW8V8HM
"""

import streamlit.components.v1 as components
import json
import os
import urllib.request
import urllib.error
import datetime

FIREBASE_CONFIG = {
    "apiKey": "AIzaSyDyS1DehH4KmEri-CzriVitQsjorIbpBbI",
    "authDomain": "ung-dung-da-nckh.firebaseapp.com",
    "projectId": "ung-dung-da-nckh",
    "storageBucket": "ung-dung-da-nckh.firebasestorage.app",
    "messagingSenderId": "628733716848",
    "appId": "1:628733716848:web:eb9732c929e874e210c7b5",
    "measurementId": "G-17SYW8V8HM"
}


def inject_firebase_analytics():
    """
    Injects Firebase Web SDK 10.8+ and Google Analytics 4 (GA4).
    Enables real-time visitor tracking and event logging for SmartResearch Studio.
    """
    tracking_html = f"""
    <div style="display: none; height: 0; width: 0;">
      <!-- Google tag (gtag.js) for Firebase / GA4 -->
      <script async src="https://www.googletagmanager.com/gtag/js?id={FIREBASE_CONFIG['measurementId']}"></script>
      <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{FIREBASE_CONFIG['measurementId']}', {{
          'page_title': 'SmartResearch Studio',
          'page_location': (window.parent && window.parent.location) ? window.parent.location.href : window.location.href
        }});

        // Try injecting directly into parent head if permitted
        try {{
          var pDoc = window.parent.document;
          if (pDoc && !pDoc.getElementById('firebase-ga-script')) {{
            var gScript = pDoc.createElement('script');
            gScript.id = 'firebase-ga-script';
            gScript.async = true;
            gScript.src = 'https://www.googletagmanager.com/gtag/js?id={FIREBASE_CONFIG['measurementId']}';
            pDoc.head.appendChild(gScript);

            var initScript = pDoc.createElement('script');
            initScript.innerHTML = "window.dataLayer = window.dataLayer || []; function gtag(){{dataLayer.push(arguments);}} gtag('js', new Date()); gtag('config', '{FIREBASE_CONFIG['measurementId']}');";
            pDoc.head.appendChild(initScript);
          }}
        }} catch(e) {{
          // Cross-origin fallback: analytics run within iframe
        }}
      </script>

      <!-- Firebase Modular JS SDK -->
      <script type="module">
        import {{ initializeApp }} from "https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js";
        import {{ getAnalytics, logEvent }} from "https://www.gstatic.com/firebasejs/10.8.0/firebase-analytics.js";

        const firebaseConfig = {{
          apiKey: "{FIREBASE_CONFIG['apiKey']}",
          authDomain: "{FIREBASE_CONFIG['authDomain']}",
          projectId: "{FIREBASE_CONFIG['projectId']}",
          storageBucket: "{FIREBASE_CONFIG['storageBucket']}",
          messagingSenderId: "{FIREBASE_CONFIG['messagingSenderId']}",
          appId: "{FIREBASE_CONFIG['appId']}",
          measurementId: "{FIREBASE_CONFIG['measurementId']}"
        }};

        try {{
          const app = initializeApp(firebaseConfig);
          const analytics = getAnalytics(app);
          logEvent(analytics, 'page_view', {{
            page_title: 'SmartResearch Studio',
            page_path: window.location.pathname
          }});
          window.firebaseApp = app;
          window.firebaseAnalytics = analytics;
        }} catch(err) {{
          console.warn("Firebase Analytics Initialization:", err);
        }}
      </script>
    </div>
    """
    components.html(tracking_html, height=0, width=0)


def send_feedback_to_firestore(name: str, email: str, role: str, topic: str, rating: int, content: str) -> tuple:
    """
    Sends user feedback to Firebase Cloud Firestore collection 'feedbacks'.
    Also creates a local backup log in 'data/feedbacks.jsonl'.
    Returns (success: bool, detail: str).
    """
    if not content or not content.strip():
        return False, "Nội dung góp ý không được để trống."

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    feedback_record = {
        "name": (name.strip() if name and name.strip() else "Người dùng ẩn danh"),
        "email": (email.strip() if email and email.strip() else "Không cung cấp"),
        "role": role or "Nhà nghiên cứu / Giảng viên",
        "topic": topic or "Đề xuất tính năng mới",
        "rating": rating,
        "content": content.strip(),
        "created_at": now_iso,
        "platform": "SmartResearch Studio Web"
    }

    # Always save a local backup copy so feedback is never lost
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        local_log_path = os.path.join(data_dir, "feedbacks.jsonl")
        with open(local_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_record, ensure_ascii=False) + "\n")
    except Exception:
        pass

    url = (
        f"https://firestore.googleapis.com/v1/projects/{FIREBASE_CONFIG['projectId']}"
        f"/databases/(default)/documents/feedbacks?key={FIREBASE_CONFIG['apiKey']}"
    )

    doc_payload = {
        "fields": {
            "name": {"stringValue": feedback_record["name"]},
            "email": {"stringValue": feedback_record["email"]},
            "role": {"stringValue": feedback_record["role"]},
            "topic": {"stringValue": feedback_record["topic"]},
            "rating": {"integerValue": str(feedback_record["rating"])},
            "content": {"stringValue": feedback_record["content"]},
            "created_at": {"timestampValue": now_iso},
            "platform": {"stringValue": feedback_record["platform"]}
        }
    }

    try:
        req_data = json.dumps(doc_payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 201):
                return True, "Gửi ý kiến đóng góp thành công!"
            return False, f"Máy chủ phản hồi mã {resp.status}."
    except urllib.error.HTTPError as err:
        err_body = err.read().decode("utf-8", errors="ignore")
        if err.code == 403:
            return False, "PERMISSION_DENIED"
        return False, f"Lỗi HTTP {err.code}: {err.reason}"
    except Exception as exc:
        return False, f"Lỗi kết nối: {str(exc)}"
