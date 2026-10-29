// Firebase configuration for SmartResearch Studio
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyDyS1DehH4KmEri-CzriVitQsjorIbpBbI",
  authDomain: "ung-dung-da-nckh.firebaseapp.com",
  projectId: "ung-dung-da-nckh",
  storageBucket: "ung-dung-da-nckh.firebasestorage.app",
  messagingSenderId: "628733716848",
  appId: "1:628733716848:web:eb9732c929e874e210c7b5",
  measurementId: "G-17SYW8V8HM"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
export const analytics = getAnalytics(app);
