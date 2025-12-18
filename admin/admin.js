import { auth, provider, db } from "./firebase.js";
import {
  signInWithPopup,
  signOut,
  onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

import {
  collection,
  addDoc,
  getDocs,
  query,
  orderBy
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// Elements
const loginBtn = document.getElementById("loginBtn");
const logoutBtn = document.getElementById("logoutBtn");
const adminSection = document.getElementById("adminSection");
const addBtn = document.getElementById("addBtn");
const videoUrlInput = document.getElementById("videoUrl");
const categorySelect = document.getElementById("category");
const statusText = document.getElementById("status");
const videoList = document.getElementById("videoList");
const jsonInput = document.getElementById("jsonInput");
const importBtn = document.getElementById("importBtn");
const jsonStatus = document.getElementById("jsonStatus");
// -------- BULK JSON IMPORT --------
importBtn.onclick = async () => {
  const raw = jsonInput.value.trim();

  if (!raw) {
    alert("Paste JSON data first");
    return;
  }

  let items;

  try {
    items = JSON.parse(raw);
  } catch (e) {
    jsonStatus.innerText = "❌ Invalid JSON format";
    return;
  }

  if (!Array.isArray(items)) {
    jsonStatus.innerText = "❌ JSON must be an array";
    return;
  }

  const allowedCategories = [
    "Kid",
    "Teen",
    "Young Adult",
    "Adult",
    "Senior"
  ];

  jsonStatus.innerText = "Importing...";

  let success = 0;
  let failed = 0;

  for (const item of items) {
    if (
      !item.url ||
      !item.category ||
      !allowedCategories.includes(item.category)
    ) {
      failed++;
      continue;
    }

    try {
      await addDoc(collection(db, "reels"), {
        url: item.url,
        category: item.category,
        createdAt: Date.now()
      });
      success++;
    } catch (err) {
      console.error(err);
      failed++;
    }
  }

  jsonStatus.innerText =
    `✅ Imported: ${success} | ❌ Failed: ${failed}`;

  if (success > 0) {
    jsonInput.value = "";
    loadVideos();
  }
};

// Login
loginBtn.onclick = async () => {
  await signInWithPopup(auth, provider);
};

// Logout
logoutBtn.onclick = async () => {
  await signOut(auth);
};

// Auth state
onAuthStateChanged(auth, user => {
  if (user) {
    loginBtn.style.display = "none";
    logoutBtn.style.display = "inline-block";
    adminSection.style.display = "block";
    loadVideos();
  } else {
    loginBtn.style.display = "inline-block";
    logoutBtn.style.display = "none";
    adminSection.style.display = "none";
  }
});

// Save video link
addBtn.onclick = async () => {
  const url = videoUrlInput.value.trim();
  const category = categorySelect.value;

  if (!url) {
    alert("Please enter a video URL");
    return;
  }

  statusText.innerText = "Saving...";

  try {
    await addDoc(collection(db, "reels"), {
      url,
      category,
      createdAt: Date.now()
    });

    statusText.innerText = "Video saved successfully ✔";
    videoUrlInput.value = "";
    loadVideos();
  } catch (err) {
    console.error(err);
    statusText.innerText = "Error saving video";
  }
};

// Load videos
async function loadVideos() {
  videoList.innerHTML = "";

  const q = query(collection(db, "reels"), orderBy("createdAt", "desc"));
  const snap = await getDocs(q);

  snap.forEach(doc => {
    const v = doc.data();
    const div = document.createElement("div");
    div.className = "video-item";
    div.innerHTML = `
      <span>${v.category}</span>
      <a href="${v.url}" target="_blank">${v.url}</a>
    `;
    videoList.appendChild(div);
  });
}
