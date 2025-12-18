import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyAhyzYmv9VzH99-3vwqQiE0ZbN2Xe2CaXg",
  authDomain: "resolute-oxygen-391422.firebaseapp.com",
  projectId: "resolute-oxygen-391422",
  storageBucket: "resolute-oxygen-391422.firebasestorage.app",
  messagingSenderId: "883503394277",
  appId: "1:883503394277:web:a764fdc212dd3dd315d30c"
};

export const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
