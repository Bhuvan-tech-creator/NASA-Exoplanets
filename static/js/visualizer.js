// Visualizer specific JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const visualizerForm = document.getElementById('visualizer-form');
    const canvas = document.getElementById('planet-canvas');
    const canvasContainer = document.getElementById('exoplanet-visualizer');
    let animationId = null;
    let isPlaying = false;
    let angle = 0; // current angle along the orbit (radians)
    let params = {
        orbitalPeriod: 365,
        planetRadius: 1.0,
        stellarTemp: 5778,
        speed: 1.0,
        eccentricity: 0.0, // Planet A
        eccentricity2: 0.0, // Planet B
        orbitDistance: 0.6, // 0..1 fraction of max A (Planet A)
        orbitDistance2: 0.5, // 0..1 fraction of max B (Planet B)
        flipA: false,
        flipB: false,
        planet2Enabled: false,
        planet2: { period: 100, radius: 2.5 }
    };
    let lastTs = null; // last timestamp for RAF
    let trail = []; // store last N positions for planet A
    let trail2 = []; // store last N positions for planet B
    const MAX_TRAIL = 120;

    // Visualizer form submission
    if (visualizerForm) {
        visualizerForm.addEventListener('submit', function(e) {
            e.preventDefault(); // Prevent page refresh
            
            // Get form values
            params.orbitalPeriod = parseFloat(document.getElementById('vis-orbital-period').value);
            params.planetRadius = parseFloat(document.getElementById('vis-planet-radius').value);
            params.stellarTemp = parseFloat(document.getElementById('vis-stellar-temp').value);

            // Update visualization immediately
            drawFrame();
        });
    }

    // Animation controls
    const playPauseBtn = document.getElementById('play-pause-btn');
    const resetBtn = document.getElementById('reset-btn');
    const speedControl = document.getElementById('speed-control');
    const toggleTransit = document.getElementById('toggle-transit');
    const toggleTrail = document.getElementById('toggle-trail');
    const toggleBrightness = document.getElementById('toggle-brightness');
    const eccControl = document.getElementById('eccentricity-control');
    const ecc2Control = document.getElementById('eccentricity2-control');
    const toggleP2 = document.getElementById('toggle-planet2');
    const p2Period = document.getElementById('p2-period');
    const p2Radius = document.getElementById('p2-radius');
    const orbitDistance = document.getElementById('orbit-distance');
    const orbitDistance2 = document.getElementById('orbit-distance2');
    const orbitFlipA = document.getElementById('orbit-flip-a');
    const orbitFlipB = document.getElementById('orbit-flip-b');

    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', function() {
            isPlaying = !isPlaying;
            this.innerHTML = isPlaying ? 
                '<i class="fas fa-pause"></i> Pause' : 
                '<i class="fas fa-play"></i> Start';
            
            if (isPlaying) {
                startAnimation();
            } else {
                stopAnimation();
            }
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', resetVisualization);
    }

    if (speedControl) {
        speedControl.addEventListener('input', function() {
            params.speed = parseFloat(this.value);
            drawFrame();
        });
    }
    if (toggleTransit) {
        toggleTransit.addEventListener('change', drawFrame);
    }
    if (toggleTrail) {
        toggleTrail.addEventListener('change', drawFrame);
    }
    if (toggleBrightness) {
        toggleBrightness.addEventListener('change', drawFrame);
    }
    if (eccControl) {
        eccControl.addEventListener('input', function() {
            params.eccentricity = parseFloat(this.value);
            drawFrame();
        });
    }
    if (ecc2Control) {
        ecc2Control.addEventListener('input', function() {
            params.eccentricity2 = parseFloat(this.value);
            drawFrame();
        });
    }
    if (toggleP2) {
        toggleP2.addEventListener('change', function() {
            params.planet2Enabled = this.checked;
            drawFrame();
        });
    }
    if (orbitDistance) {
        orbitDistance.addEventListener('input', function() {
            params.orbitDistance = parseFloat(this.value);
            drawFrame();
        });
    }
    if (orbitDistance2) {
        orbitDistance2.addEventListener('input', function() {
            params.orbitDistance2 = parseFloat(this.value);
            drawFrame();
        });
    }
    if (orbitFlipA) {
        orbitFlipA.addEventListener('change', function() {
            params.flipA = this.checked;
            drawFrame();
        });
    }
    if (orbitFlipB) {
        orbitFlipB.addEventListener('change', function() {
            params.flipB = this.checked;
            drawFrame();
        });
    }
    if (p2Period) {
        p2Period.addEventListener('input', function() {
            params.planet2.period = Math.max(1, parseFloat(this.value)||100);
            drawFrame();
        });
    }
    if (p2Radius) {
        p2Radius.addEventListener('input', function() {
            params.planet2.radius = Math.max(0.1, parseFloat(this.value)||2.5);
            drawFrame();
        });
    }

    // Visualization functions
    function drawFrame() {
        if (!canvas || !canvas.getContext) return;
        
    const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate star color based on temperature
        const starColor = getStarColor(params.stellarTemp);
        
        // Draw star
        ctx.beginPath();
        ctx.fillStyle = starColor;
        ctx.arc(canvas.width / 2, canvas.height / 2, 50, 0, Math.PI * 2);
        ctx.fill();
        // Star glow scales lightly with temperature
        ctx.save();
        ctx.shadowColor = starColor;
        ctx.shadowBlur = Math.min(50, Math.max(10, (params.stellarTemp - 3000) / 50));
        ctx.beginPath();
        ctx.arc(canvas.width / 2, canvas.height / 2, 50, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

    // Orbit sizing and clamping so everything stays in view
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const STAR_R = 50;
    const MARGIN = 30;
    const leftRoom = cx - MARGIN;
    const rightRoom = canvas.width - cx - MARGIN;
    const topRoom = cy - MARGIN;
    const bottomRoom = canvas.height - cy - MARGIN;
    const maxVertRoom = Math.min(topRoom, bottomRoom);
    const e = Math.min(0.9, Math.max(0, params.eccentricity));
    const e2 = Math.min(0.9, Math.max(0, params.eccentricity2));
    const maxAByRight = rightRoom / (1 + e);
    const maxAByVert = maxVertRoom / Math.max(0.0001, Math.sqrt(1 - e * e));
    const maxA = Math.max(10, Math.min(maxAByRight, maxAByVert));
    // Planet A semi-major axis controlled by slider (fraction of max)
    const a = params.orbitDistance * maxA;
    const b = a * Math.sqrt(1 - e*e); // semi-minor axis
        
        // Draw orbit (ellipse)
        ctx.beginPath();
        ctx.strokeStyle = '#666';
        // ellipse centered at (cx + focusOffset, cy) with radii a,b
        if (ctx.ellipse) {
            const focusOffsetA = (params.flipA ? -1 : 1) * (a * e);
            ctx.ellipse(cx + focusOffsetA, cy, a, b, 0, 0, Math.PI * 2);
            ctx.stroke();
        } else {
            // Fallback: approximate ellipse with many line segments
            ctx.beginPath();
            for (let t = 0; t <= Math.PI * 2 + 0.01; t += 0.05) {
                const focusOffsetA = (params.flipA ? -1 : 1) * (a * e);
                const ex = cx + focusOffsetA + a * Math.cos(t);
                const ey = cy + b * Math.sin(t);
                if (t === 0) ctx.moveTo(ex, ey); else ctx.lineTo(ex, ey);
            }
            ctx.closePath();
            ctx.stroke();
        }

        // Planet A position at current angle
        // Elliptical orbit parameterization (centered on star with focus offset)
    const focusOffset = (params.flipA ? -1 : 1) * (a * e); // distance from center to focus (directional)
    const x = cx + focusOffset + a * Math.cos(angle);
        const y = cy + b * Math.sin(angle);
        const planetSize = 5 + (params.planetRadius * 2);

        // Draw A's body now so trail shows underneath
        ctx.beginPath();
        ctx.fillStyle = '#3498db';
        ctx.arc(x, y, planetSize, 0, Math.PI * 2);
        ctx.fill();

        // Draw a simple radial velocity wobble line (star's tiny motion)
        // Combined wobble from planet(s)
        let wobbleAmp = Math.min(4, params.planetRadius * 0.4);
        let wobblePhase = angle + Math.PI;
        if (params.planet2Enabled) {
            const ang2 = (angle * params.orbitalPeriod) / Math.max(1, params.planet2.period);
            // scale wobble more for shorter period (closer-in planet)
            wobbleAmp += Math.min(4, params.planet2.radius * 0.2) * (365 / Math.max(30, params.planet2.period));
            // simple phase mix
            wobblePhase = (wobblePhase + ang2) / 2;
        }
        const wobbleX = cx + wobbleAmp * Math.cos(wobblePhase);
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(90, 139, 212, 0.6)';
        ctx.moveTo(cx - 10, cy);
        ctx.lineTo(cx + 10, cy);
        ctx.moveTo(cx, cy);
        ctx.lineTo(wobbleX, cy);
        ctx.stroke();

        // Planet A trail
        if (toggleTrail && toggleTrail.checked) {
            trail.push({ x, y, r: planetSize });
            if (trail.length > MAX_TRAIL) trail.shift();
            ctx.save();
            for (let i = 0; i < trail.length; i++) {
                const t = trail[i];
                const alpha = i / trail.length;
                ctx.fillStyle = `rgba(52, 152, 219, ${alpha * 0.5})`;
                ctx.beginPath();
                ctx.arc(t.x, t.y, Math.max(1, t.r * 0.6), 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.restore();
        } else {
            trail = [];
        }

        // Planet B trail and optional drawing
        let x2, y2, r2;
        if (params.planet2Enabled) {
            // Planet B: compute size from period, scale by distance fraction, clamp to view using its own eccentricity e2
            let a2Natural = 120 + 30 * Math.log10(Math.max(1, params.planet2.period));
            const maxA2ByRight = rightRoom / (1 + e2);
            const maxA2ByVert = maxVertRoom / Math.max(0.0001, Math.sqrt(1 - e2 * e2));
            const maxA2 = Math.max(10, Math.min(maxA2ByRight, maxA2ByVert));
            let a2 = Math.min(a2Natural, maxA2) * params.orbitDistance2;
            const b2 = a2 * Math.sqrt(1 - e2*e2);
            const ang2 = (angle * params.orbitalPeriod) / Math.max(1, params.planet2.period);
            const focusOffsetB = (params.flipB ? -1 : 1) * (a2 * e2);
            x2 = cx + focusOffsetB + a2 * Math.cos(ang2);
            y2 = cy + b2 * Math.sin(ang2);
            r2 = 5 + (params.planet2.radius * 2);
            if (toggleTrail && toggleTrail.checked) {
                trail2.push({ x: x2, y: y2, r: r2 });
                if (trail2.length > MAX_TRAIL) trail2.shift();
                ctx.save();
                for (let i = 0; i < trail2.length; i++) {
                    const t = trail2[i];
                    const alpha = i / trail2.length;
                    ctx.fillStyle = `rgba(46, 204, 113, ${alpha * 0.5})`;
                    ctx.beginPath();
                    ctx.arc(t.x, t.y, Math.max(1, t.r * 0.6), 0, Math.PI * 2);
                    ctx.fill();
                }
                ctx.restore();
            } else {
                trail2 = [];
            }
            // Draw B body on top of its trail
            ctx.beginPath();
            ctx.fillStyle = '#2ecc71';
            ctx.arc(x2, y2, r2, 0, Math.PI * 2);
            ctx.fill();
        }

        // Planet labels (A/B)
        drawPlanetLabel(ctx, x, y, planetSize, '#3498db', 'A');
        if (params.planet2Enabled && x2 !== undefined) {
            drawPlanetLabel(ctx, x2, y2, r2, '#2ecc71', 'B');
        }

        // Transit dimming: consider both planets
        if (toggleTransit && toggleTransit.checked) {
            let dim = 0;
            // primary planet
            let inFront = Math.abs(y - cy) < planetSize && x < cx;
            if (inFront) dim += Math.min(0.25, (planetSize / 60) ** 2);
            // second planet
            if (params.planet2Enabled) {
                // use x2,y2,r2 from above
                if (Math.abs(y2 - cy) < r2 && x2 < cx) dim += Math.min(0.25, (r2 / 60) ** 2);
            }
            if (dim > 0) {
                ctx.save();
                ctx.fillStyle = `rgba(0,0,0,${dim})`;
                ctx.beginPath();
                ctx.arc(cx, cy, 55, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            }
        }

        // Brightness meter (proxy): shows 100% - transit depth-like dip from both planets
        if (toggleBrightness && toggleBrightness.checked) {
            const meterW = 160, meterH = 10;
            const meterX = canvas.width - meterW - 20;
            const meterY = 20;
            ctx.save();
            ctx.fillStyle = 'rgba(255,255,255,0.2)';
            ctx.fillRect(meterX, meterY, meterW, meterH);
            let brightness = 1.0;
            // primary planet dip
            let inFront = Math.abs(y - cy) < planetSize && x < cx;
            if (inFront) brightness -= Math.min(0.05, (planetSize / 60) ** 2);
            // second planet dip
            if (params.planet2Enabled) {
                // use x2,y2,r2 from above
                if (Math.abs(y2 - cy) < r2 && x2 < cx) brightness -= Math.min(0.05, (r2 / 60) ** 2);
            }
            ctx.fillStyle = '#5a8bd4';
            ctx.fillRect(meterX, meterY, meterW * brightness, meterH);
            ctx.fillStyle = 'rgba(255,255,255,0.7)';
            ctx.font = '12px Arial';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(`Brightness: ${(brightness * 100).toFixed(0)}%`, meterX + meterW, meterY + meterH/2);
            ctx.restore();
        }

        // Update fun facts text
        const facts = document.getElementById('viz-facts');
        if (facts) {
            let desc = 'Tip: Try a Hot Jupiter preset for fast orbits!';
            if (params.orbitalPeriod < 10) desc = 'Hot Jupiters orbit in just a few days. Super fast!';
            else if (params.planetRadius > 8) desc = 'Big planets can make stars wobble more (radial velocity)!';
            else if (params.stellarTemp > 6500) desc = 'Hotter (bluer) stars shine brighter and bluer!';
            else if (params.orbitalPeriod > 1000) desc = 'Long periods mean large orbits—farther from the star.';
            if (e > 0.4) desc = 'High eccentricity = very elongated orbit!';
            if (params.planet2Enabled) desc = 'Two planets: combined wobble and multiple transits possible!';
            facts.textContent = desc;
        }
    }

    function startAnimation() {
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
        lastTs = null;
        animationId = requestAnimationFrame(animate);
    }

    function stopAnimation() {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    }

    function animate(ts) {
        if (!lastTs) lastTs = ts;
        const dt = (ts - lastTs) / 1000; // seconds
        lastTs = ts;

        // Angular speed proportional to 1 / orbitalPeriod
        const baseAngularSpeed = (2 * Math.PI) / Math.max(params.orbitalPeriod, 0.1); // rad per day (unit)
        const speedScale = 120; // scale up for visual effect
        angle += baseAngularSpeed * params.speed * speedScale * dt;
        angle %= (Math.PI * 2);

        drawFrame();
        animationId = requestAnimationFrame(animate);
    }

    function resetVisualization() {
        stopAnimation();
        params = {
            orbitalPeriod: 365,
            planetRadius: 1.0,
            stellarTemp: 5778,
            speed: 1.0,
            eccentricity: 0.0,
            eccentricity2: 0.0,
            orbitDistance: 0.6,
            orbitDistance2: 0.5,
            flipA: false,
            flipB: false,
            planet2Enabled: false,
            planet2: { period: 100, radius: 2.5 }
        };
        angle = 0;
        // Reset form values
        document.getElementById('vis-orbital-period').value = params.orbitalPeriod;
        document.getElementById('vis-planet-radius').value = params.planetRadius;
        document.getElementById('vis-stellar-temp').value = params.stellarTemp;
        if (speedControl) speedControl.value = params.speed;
        if (eccControl) eccControl.value = params.eccentricity;
        if (ecc2Control) ecc2Control.value = params.eccentricity2;
        if (toggleP2) toggleP2.checked = params.planet2Enabled;
        if (p2Period) p2Period.value = params.planet2.period;
        if (p2Radius) p2Radius.value = params.planet2.radius;
    if (orbitDistance) orbitDistance.value = params.orbitDistance;
    if (orbitDistance2) orbitDistance2.value = params.orbitDistance2;
    if (orbitFlipA) orbitFlipA.checked = params.flipA;
    if (orbitFlipB) orbitFlipB.checked = params.flipB;
        
        drawFrame();
    }

    function drawPlanetLabel(ctx, x, y, r, color, text) {
        const offset = Math.max(14, r + 8);
        const lx = x + offset;
        const ly = y - offset;
        const padX = 6, padY = 3;
        ctx.font = 'bold 12px Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
        const metrics = ctx.measureText(text);
        const bw = Math.ceil(metrics.width) + padX * 2;
        const bh = 18;
        // Background
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        roundRect(ctx, lx - bw / 2, ly - bh / 2, bw, bh, 6);
        ctx.fill();
        // Border
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        roundRect(ctx, lx - bw / 2, ly - bh / 2, bw, bh, 6);
        ctx.stroke();
        // Text
        ctx.fillStyle = '#eaf2ff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, lx, ly);
    }

    function roundRect(ctx, x, y, w, h, r) {
        const radius = Math.min(r, w / 2, h / 2);
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + w - radius, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
        ctx.lineTo(x + w, y + h - radius);
        ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
        ctx.lineTo(x + radius, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    }

    // speed handled in input listener

    function getStarColor(temp) {
        // Simple temperature to color conversion
        if (temp < 3500) return '#ff6b3d';  // Cool red stars
        if (temp < 5000) return '#ffb347';  // Orange stars
        if (temp < 6000) return '#ffe066';  // Yellow stars like our Sun
        if (temp < 7500) return '#d6f0ff';  // White stars
        return '#a7d3ff';  // Hot blue stars
    }

    // Canvas responsive sizing
    function resizeCanvas() {
        if (!canvas || !canvasContainer) return;
        const rect = canvasContainer.getBoundingClientRect();
        const width = Math.max(640, Math.floor(rect.width));
        const height = Math.max(420, Math.floor(width * 0.56));
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        drawFrame();
    }

    // Initialize visualization with default values and resize listeners
    resetVisualization();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Preset cards support (if present on page)
    document.querySelectorAll('.preset-card').forEach(card => {
        card.addEventListener('click', () => {
            const preset = card.dataset.preset;
            const presets = {
                earth: { orbitalPeriod: 365, planetRadius: 1.0, stellarTemp: 5778 },
                jupiter: { orbitalPeriod: 4333, planetRadius: 11.2, stellarTemp: 5778 },
                hotjupiter: { orbitalPeriod: 3, planetRadius: 10.0, stellarTemp: 6000 },
                superearth: { orbitalPeriod: 100, planetRadius: 2.5, stellarTemp: 5500 }
            };
            const p = presets[preset];
            if (p) {
                params = { ...params, ...p };
                // Update form fields if present
                const f1 = document.getElementById('vis-orbital-period'); if (f1) f1.value = params.orbitalPeriod;
                const f2 = document.getElementById('vis-planet-radius'); if (f2) f2.value = params.planetRadius;
                const f3 = document.getElementById('vis-stellar-temp'); if (f3) f3.value = params.stellarTemp;
                drawFrame();
            }
        });
    });
});