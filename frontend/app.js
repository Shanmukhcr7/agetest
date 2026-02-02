import { db, auth, googleProvider, signInWithPopup, signOut, onAuthStateChanged } from "./firebase.js";
import { collection, getDocs, query, where, limit, startAfter, orderBy } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
    API_URL: "http://localhost:8000/api/age-check",
    SAFETY: {
        CONFIDENCE_THRESHOLD: 0.60,
        ADULT_FRAME_BUFFER: 1, // Unlock faster (20s interval is long enough)
        KID_LOCK_IMMEDIATE: true
    },
    INTERVAL_MS: 20000 // 20 Seconds
};

// ============================================
// ROUTER (VIEW MANAGER)
// ============================================
class Router {
    constructor(app) {
        this.app = app;
        this.views = document.querySelectorAll(".view");
        this.navItems = document.querySelectorAll(".nav-item");

        this.currentView = "login"; // Default
        this.scrollStore = {};

        // Nav Click
        this.navItems.forEach(item => {
            item.addEventListener("click", () => {
                const target = item.dataset.target;
                if (target) this.navigate(target);
            });
        });

        // Hide Nav on Login
        this.navBar = document.querySelector(".bottom-nav");
    }

    navigate(route) {
        if (route === this.currentView) return;

        // Guard: Auth
        if (!auth.currentUser && route !== "login") {
            console.warn("Auth Locked. Redirecting to Login.");
            route = "login";
        }

        // Save Scroll of current view
        const currentEl = document.getElementById(`view-${this.currentView}`);
        if (currentEl) this.scrollStore[this.currentView] = currentEl.scrollTop;

        // Update State
        this.currentView = route;

        // 1. Update Views
        this.views.forEach(view => {
            if (view.id === `view-${route}`) {
                view.classList.add("active-view");
                gsap.fromTo(view, { opacity: 0, x: 20 }, { opacity: 1, x: 0, duration: 0.4 });
                // Trigger On-Load events for views
                if (route === "reels") this.app.reelManager.play(); // Play focused
                if (route === "home") this.app.homeManager.loadSections(this.app.safetyEngine.currentMode, true); // Pass current mode and reset
                if (route === "profile") this.app.profileManager.updateUI();

                // Restore Scroll
                if (this.scrollStore[route] !== undefined) {
                    setTimeout(() => view.scrollTop = this.scrollStore[route], 50); // Small delay for rendering
                }
            } else {
                view.classList.remove("active-view");
                // Pause Reels if leaving
                if (view.id === "view-reels") this.app.reelManager.pause();
            }
        });

        // 2. Update Nav State
        this.navItems.forEach(item => {
            if (item.dataset.target === route) item.classList.add("active-nav");
            else item.classList.remove("active-nav");
        });

        // 3. Show/Hide Nav Bar
        if (route === "login") {
            gsap.to(this.navBar, { y: 100, opacity: 0, duration: 0.3 });
        } else {
            gsap.to(this.navBar, { y: 0, opacity: 1, duration: 0.3 });
        }
    }
}

// ============================================
// MANAGERS
// ============================================

class ProfileManager {
    constructor() {
        this.nameEl = document.getElementById("profileName");
        this.emailEl = document.getElementById("profileEmail");
        this.avatarEl = document.getElementById("profileAvatar");
        this.btnLogout = document.getElementById("btnLogout");

        if (this.btnLogout) {
            this.btnLogout.addEventListener("click", () => {
                signOut(auth).then(() => window.location.reload());
            });
        }
    }

    updateUI() {
        const user = auth.currentUser;
        if (user) {
            this.nameEl.innerText = user.displayName;
            this.emailEl.innerText = user.email;
            if (user.photoURL) {
                this.avatarEl.innerHTML = `<img src="${user.photoURL}" alt="Profile">`;
            }
        }
    }
}





class HomeManager {
    constructor(app) {
        this.app = app;
        this.grid = document.getElementById("homeGrid");
        this.loading = false;
        // Remove class content-grid to allow custom layout
        this.grid.className = "home-content";
        this.currentMode = "Kid"; // Default
    }

    async loadSections(tag = null, reset = false) {
        if (tag) this.currentTag = tag;
        if (this.loading) return;
        this.loading = true;
        if (reset) {
            this.grid.innerHTML = "";
        }

        try {
            // Define categories based on specific tag
            let categories = ["Kid", "kid"]; // Default Safe

            // Map Tags to Categories
            const tagMap = {
                "Kid": ["Kid", "kid"],
                "Teen": ["Teen", "Kid", "teen"],
                "Young Adult": ["Young Adult", "Teen", "Adult", "adult"],
                "Adult": ["Adult", "Senior", "Young Adult", "adult", "teen"],
                "Senior": ["Adult", "Senior", "Young Adult", "adult"]
            };

            if (tagMap[this.currentTag]) {
                categories = tagMap[this.currentTag];
            }

            console.log(`HOME: Loading for TAG: ${this.currentTag}, Categories:`, categories);

            // Get Videos (Limit 20 for sections)
            const q = query(
                collection(db, "reels"),
                where("category", "in", categories),
                limit(20)
            );
            const snapshot = await getDocs(q);
            const videos = [];
            snapshot.forEach(doc => videos.push(doc.data()));

            console.log(`HOME: Found ${videos.length} videos`);

            // Split into sections
            if (videos.length > 0) {
                const trending = videos.slice(0, 10);
                const liked = videos.slice(10, 20).length ? videos.slice(10, 20) : videos.slice(0, 5); // Fallback

                this.renderSection(`For ${this.currentTag}s`, trending, "trending");
                this.renderSection("❤️ Most Liked", liked, "liked");
            } else {
                this.renderFallback();
            }

        } catch (e) {
            console.error("Home Load Error", e);
            if (reset) this.renderFallback();
        }
        this.loading = false;
    }

    renderSection(title, videos, type) {
        // ... (Keep existing renderSection)
        // Section Container
        const section = document.createElement("div");
        section.className = "home-section";

        // Title
        const heading = document.createElement("h3");
        heading.className = "section-title";
        heading.innerHTML = type === 'trending' ? `<i data-lucide="flame"></i> ${title}` : `<i data-lucide="heart"></i> ${title}`;
        section.appendChild(heading);

        // Scroll Container
        const scrollRow = document.createElement("div");
        scrollRow.className = "horizontal-scroll";

        videos.forEach(data => {
            this.renderCard(data, scrollRow);
        });

        section.appendChild(scrollRow);
        this.grid.appendChild(section);
        if (window.lucide) lucide.createIcons();
    }

    // ... (Keep existing renderCard, extractVideoId, renderFallback)
    renderCard(data, container) {
        const videoId = this.extractVideoId(data.url);
        if (!videoId) return;

        // Mock Stats
        const views = Math.floor(Math.random() * 900) + 100 + "K";
        const likes = (Math.random() * 9).toFixed(1) + "K";

        const card = document.createElement("div");
        card.className = "card-item glass-panel";
        card.innerHTML = `
            <img src="https://img.youtube.com/vi/${videoId}/hqdefault.jpg" loading="lazy">
            <div class="card-tag">HD</div>
            <div class="card-overlay">
                <div style="font-weight:600; font-size: 0.9rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    ${data.title || "Watch Video"}
                </div>
                <div class="card-stats">
                    <span><i data-lucide="eye" style="width:12px"></i> ${views}</span>
                    <span><i data-lucide="thumbs-up" style="width:12px"></i> ${likes}</span>
                </div>
            </div>
        `;

        card.addEventListener("click", () => {
            if (this.app.reelManager) this.app.reelManager.playVideoId(videoId);
            this.app.router.navigate("reels");
        });

        container.appendChild(card);
    }

    extractVideoId(url) {
        let id = "";
        if (url.includes("shorts/")) id = url.split("shorts/")[1].split("?")[0];
        else if (url.includes("v=")) id = url.split("v=")[1].split("&")[0];
        else if (url.includes("youtu.be/")) id = url.split("youtu.be/")[1];
        return id;
    }

    renderFallback() {
        const mocks = [
            { url: "https://youtube.com/watch?v=5qap5aO4i9A", title: "Lo-Fi Beats" },
            { url: "https://youtube.com/watch?v=tgbNymZ7vqY", title: "Coding Life" },
            { url: "https://youtube.com/watch?v=jfKfPfyJRdk", title: "Relaxing View" }
        ];
        if (this.grid.children.length === 0) {
            this.renderSection("Recommended", mocks, "trending");
        }
    }

    // Alias for compatibility
    loadGrid(tag, reset) { this.loadSections(tag, reset); }
}

// ... SAFETY & REEL LOGIC Same as before, but wrapped ...

class ReelManager {
    constructor(app) {
        this.app = app;
        this.container = document.getElementById("reelContainer");
        this.container.innerHTML = ""; // Clear existing

        // Observer for auto-play
        this.observer = new IntersectionObserver(this.handleScroll.bind(this), {
            root: this.container,
            threshold: 0.6
        });
        this.playingFrame = null;
    }

    // ... (Keep handleScroll, playVideoId, pause, play, postMessage, extractId, etc.)
    handleScroll(entries) {
        entries.forEach(entry => {
            const iframe = entry.target.querySelector("iframe");
            if (!iframe) return;

            if (entry.isIntersecting) {
                this.postMessage(iframe, "playVideo");
                this.playingFrame = iframe;
            } else {
                this.postMessage(iframe, "pauseVideo");
            }
        });
    }

    playVideoId(id) {
        const target = Array.from(this.container.children).find(div => div.dataset.id === id);
        if (target) {
            target.scrollIntoView({ behavior: "smooth" });
        }
    }

    pause() {
        if (this.playingFrame) this.postMessage(this.playingFrame, "pauseVideo");
    }

    play() {
        if (this.playingFrame) this.postMessage(this.playingFrame, "playVideo");
    }

    async loadReels(tag) {
        console.log(`REELS: Loading for TAG: ${tag}...`);

        let categories = ["Kid", "kid"];
        const tagMap = {
            "Kid": ["Kid", "kid"],
            "Teen": ["Teen", "Kid", "teen"],
            "Young Adult": ["Young Adult", "Teen", "Adult", "adult"],
            "Adult": ["Adult", "Senior", "Young Adult", "adult", "teen"],
            "Senior": ["Adult", "Senior", "Young Adult", "adult", "senior"]
        };

        if (tagMap[tag]) categories = tagMap[tag];

        try {
            const q = query(collection(db, "reels"), where("category", "in", categories));
            const snapshot = await getDocs(q);
            const videos = [];
            snapshot.forEach(doc => {
                const d = doc.data();
                if (d.url) {
                    const id = this.extractId(d.url);
                    if (id) videos.push(id);
                }
            });

            console.log(`REELS: Found ${videos.length} videos`);
            if (videos.length === 0) this.reels = this.getFallbackReels(tag);
            else this.reels = this.shuffle(videos);

            this.renderVideos(this.reels);

        } catch (e) {
            console.warn("REELS: error", e);
            this.reels = this.getFallbackReels(tag);
            this.renderVideos(this.reels);
        }
    }

    renderVideos(list) {
        this.container.innerHTML = "";
        this.observer.disconnect();

        list.forEach(id => {
            const page = document.createElement("div");
            page.className = "reel-page";
            page.dataset.id = id;
            // Embed: loop=1 playlist=id playsinline
            const src = `https://www.youtube-nocookie.com/embed/${id}?enablejsapi=1&controls=0&rel=0&loop=1&playlist=${id}&playsinline=1`;
            page.innerHTML = `<iframe src="${src}" allow="autoplay; encrypted-media" frameborder="0" loading="lazy"></iframe>`;

            this.container.appendChild(page);
            this.observer.observe(page);
        });
    }

    postMessage(iframe, func) {
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage(JSON.stringify({ event: 'command', func: func }), '*');
        }
    }

    extractId(url) {
        let id = "";
        if (url.includes("shorts/")) id = url.split("shorts/")[1].split("?")[0];
        else if (url.includes("v=")) id = url.split("v=")[1].split("&")[0];
        else if (url.includes("youtu.be/")) id = url.split("youtu.be/")[1];
        else if (url.length === 11) id = url;
        return id;
    }

    formatUrl(url) { return url; }
    shuffle(array) { return array.sort(() => Math.random() - 0.5); }
    getFallbackReels(tag) {
        const safe = ["5qap5aO4i9A", "tgbNymZ7vqY"];
        const adult = [...safe, "jfKfPfyJRdk"];
        return (tag === "Kid" || tag === "Teen") ? safe : adult;
    }
}

class SafetyEngine {
    constructor(onAgeUpdate) {
        this.currentMode = "Kid";
        this.currentTag = "Kid";
        this.consecutiveAdultFrames = 0;
        this.errorBuffer = 0; // Grace period for missing faces
        this.MAX_ERROR_BUFFER = 3; // Allow 3 frames (~4.5s) of "No Face" before locking
        this.onAgeUpdate = onAgeUpdate;
    }

    processResult(data) {
        console.log("Detection:", data.age_group, data.confidence);
        // 1. Handle Errors (No Face, Network Error, etc)
        if (data.error || data.forced_safety) {
            console.warn("Safety Warning:", data.msg || "Unknown Error");

            if (this.currentMode !== "Kid") { // Only degrade if not already Kid
                this.errorBuffer++;
                if (this.errorBuffer > this.MAX_ERROR_BUFFER) {
                    this.forceSafeMode("Face Lost / Error Limit Reached");
                }
            }
            return;
        }

        // Reset Error Buffer on any valid detection
        this.errorBuffer = 0;

        const detectedTag = data.age_group;
        const isSafeGroup = ["Kid", "Teen"].includes(detectedTag);

        if (isSafeGroup) {
            // CRITICAL: Immediate Downgrade on Child Detection
            if (this.currentTag !== detectedTag || this.currentMode !== "Kid") {
                // Even if switching Teen -> Kid, we update.
                this.forceSafeMode(`Child Detected (${detectedTag})`, detectedTag);
            }
            this.consecutiveAdultFrames = 0;
        } else {
            // Adult Detected (Young Adult, Adult, Senior)
            if (this.currentMode === "Kid") {
                // Must pass confidence threshold to UNLOCK
                if (data.confidence >= CONFIG.SAFETY.CONFIDENCE_THRESHOLD) {
                    this.consecutiveAdultFrames++;
                    if (this.consecutiveAdultFrames >= CONFIG.SAFETY.ADULT_FRAME_BUFFER) {
                        this.unlockAdultMode(detectedTag);
                    }
                } else {
                    this.consecutiveAdultFrames = 0;
                }
            } else {
                // Already unlocked, just checking for TAG updates
                if (this.currentTag !== detectedTag) {
                    // Update tag immediately if confidence is okay
                    if (data.confidence >= CONFIG.SAFETY.CONFIDENCE_THRESHOLD) {
                        this.updateTag(detectedTag);
                    }
                }
            }
        }
    }

    forceSafeMode(reason, tag = "Kid") {
        this.currentMode = "Kid";
        this.currentTag = tag;
        this.consecutiveAdultFrames = 0;
        this.errorBuffer = 0;
        this.onAgeUpdate(tag, "Kid", reason);
    }

    unlockAdultMode(tag) {
        if (this.currentMode !== "Adult") {
            this.currentMode = "Adult";
            this.currentTag = tag;
            this.onAgeUpdate(tag, "Adult", "Verified");
        }
    }

    updateTag(tag) {
        this.currentTag = tag;
        this.onAgeUpdate(tag, "Adult", "Tag Update");
    }
}

// ============================================
// MAIN APP CONTROLLER
// ============================================
class App {
    constructor() {
        this.videoElement = document.getElementById("cameraFeed");
        this.canvas = document.getElementById("processingCanvas");
        this.ctx = this.canvas.getContext("2d");

        this.ui = {
            badge: document.getElementById("statusBadge"),
            label: document.getElementById("ageLabel"),
            splash: document.getElementById("splashScreen"),
            appContainer: document.querySelector(".app-container"),
            overlay: document.getElementById("noFaceOverlay"),
            scanOverlay: document.getElementById("scanningOverlay")
        };

        // Managers
        this.reelManager = new ReelManager(this);
        this.homeManager = new HomeManager(this);
        this.profileManager = new ProfileManager();
        this.router = new Router(this);
        this.safetyEngine = new SafetyEngine(this.handleAgeUpdate.bind(this));

        this.init();
    }

    // ... (Keep existing init, showScanning, animateIntro, startAgeDetection, captureFrame)
    async init() {
        this.animateIntro();
        // Camera
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
            this.videoElement.srcObject = stream;
        } catch (e) { console.error("Camera denied", e); }
        this.startAgeDetection();
        onAuthStateChanged(auth, (user) => { if (user) this.router.navigate("reels"); else this.router.navigate("login"); });
        document.getElementById("btnLogin").addEventListener("click", () => { signInWithPopup(auth, googleProvider).catch(err => alert(err.message)); });
        const btnLogout = document.getElementById("btnLogout");
        if (btnLogout) btnLogout.addEventListener("click", () => { signOut(auth).catch((error) => console.error("Sign Out Error", error)); });
        const btnToggle = document.getElementById("btnCameraToggle");
        if (btnToggle) btnToggle.addEventListener("click", () => this.toggleCameraVisibility());
        this.setupGestures();
        this.showScanning(true);
    }

    showScanning(show) {
        if (this.ui.scanOverlay) {
            if (show) this.ui.scanOverlay.classList.add("scanning-visible");
            else this.ui.scanOverlay.classList.remove("scanning-visible");
        }
    }

    animateIntro() {
        const tl = gsap.timeline();
        tl.to(".splash-logo", { opacity: 1, y: 0, duration: 1 })
            .to(".loader-line", { width: "100%", duration: 1 })
            .to(this.ui.splash, { opacity: 0, duration: 0.5, onComplete: () => this.ui.splash.style.display = "none" })
            .to(this.ui.appContainer, { opacity: 1, duration: 0.5 });
    }

    startAgeDetection() {
        // Define the detection routine
        const runDetection = async () => {
            // ... existing detection logic
            if (this.videoElement.readyState === 4) {
                const imgData = this.captureFrame();
                if (imgData) {
                    try {
                        const payload = this.encryptPayload(imgData);
                        const response = await fetch(CONFIG.API_URL, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(payload)
                        });
                        const data = await response.json();
                        this.safetyEngine.processResult(data);
                    } catch (e) {
                        this.safetyEngine.forceSafeMode("Network Error");
                    }
                }
            }
        };

        // 1. Initial Scan Logic
        const initScan = async () => {
            if (this.videoElement.readyState !== 4) {
                setTimeout(initScan, 500);
                return;
            }

            try {
                const imgData = this.captureFrame();
                if (imgData) {
                    const payload = this.encryptPayload(imgData);

                    const response = await fetch(CONFIG.API_URL, {
                        method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });
                    const data = await response.json();

                    // Unlock UI
                    this.showScanning(false);

                    // FAST TRACK
                    const isAdult = !["Kid", "Teen"].includes(data.age_group);
                    const isConfident = data.confidence >= CONFIG.SAFETY.CONFIDENCE_THRESHOLD;

                    if (isAdult && isConfident) {
                        this.safetyEngine.currentMode = "Adult"; // Force state
                        this.handleAgeUpdate("Adult", "Adult", "Scan Complete (Fast Track)");
                    } else {
                        this.safetyEngine.processResult(data);
                        this.handleAgeUpdate(this.safetyEngine.currentTag || "Kid", this.safetyEngine.currentMode, "Scan Complete");
                    }

                    setInterval(runDetection, CONFIG.INTERVAL_MS);
                } else {
                    setTimeout(initScan, 500);
                }
            } catch (e) {
                console.error("Init Scan Error", e);
                setTimeout(initScan, 1000); // Retry on error
            }
        };

        // Start waiting for camera
        setTimeout(initScan, 1500);
    }

    encryptPayload(base64Image) {
        // Simple Fixed Key for Demo/Dev (In prod, use session keys)
        // 32-byte key (256-bit)
        const key = CryptoJS.enc.Utf8.parse('12345678901234567890123456789012');
        const iv = CryptoJS.lib.WordArray.random(16);

        const encrypted = CryptoJS.AES.encrypt(base64Image, key, {
            iv: iv,
            mode: CryptoJS.mode.CBC,
            padding: CryptoJS.pad.Pkcs7
        });

        return {
            encrypted_data: encrypted.toString(),
            iv: iv.toString(CryptoJS.enc.Base64)
        };
    }

    captureFrame() {
        // Use reasonable resolution for Face Detection
        this.canvas.width = 480;
        this.canvas.height = 480;
        this.ctx.drawImage(this.videoElement, 0, 0, 480, 480);
        // Compress to 0.7 quality JPEG
        return this.canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
    }

    toggleCameraVisibility() {
        const wrapper = document.querySelector(".camera-wrapper");
        const btn = document.getElementById("btnCameraToggle");
        if (wrapper) { wrapper.classList.toggle("camera-hidden"); const isHidden = wrapper.classList.contains("camera-hidden"); if (btn) { btn.innerHTML = isHidden ? `<i data-lucide="video-off"></i>` : `<i data-lucide="video"></i>`; if (window.lucide) lucide.createIcons(); } }
    }

    setupGestures() { if (window.lucide) window.lucide.createIcons(); }

    handleAgeUpdate(tag, mode, reason) {
        console.log(`UPDATE: ${tag} (${mode}) - ${reason}`);

        // UI Updates
        if (mode === "Kid" || tag === "Teen") {
            // Safe Mode UI
            this.ui.badge.className = "status-badge status-kid";
            // Check specifically for Face Lost to show overlay
            if (reason && reason.includes("Face Lost")) {
                if (this.ui.overlay) this.ui.overlay.classList.add("no-face-visible");
                this.reelManager.pause();
                return;
            }
        } else {
            // Adult Mode UI
            this.ui.badge.className = "status-badge status-adult";
        }

        // Update Label with EXACT TAG
        this.ui.label.innerText = tag.toUpperCase();

        // Resume if valid
        if (this.ui.overlay) this.ui.overlay.classList.remove("no-face-visible");

        // Reload Content with specific TAG
        this.reelManager.loadReels(tag);
        this.homeManager.loadGrid(tag, true);
    }
}

window.addEventListener("load", () => new App());

