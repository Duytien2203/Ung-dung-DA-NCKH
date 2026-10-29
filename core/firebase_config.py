"""
Firebase & Google Analytics Integration for SmartResearch Studio
Configured for project: ung-dung-da-nckh
Measurement ID: G-17SYW8V8HM
"""

import streamlit.components.v1 as components

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
