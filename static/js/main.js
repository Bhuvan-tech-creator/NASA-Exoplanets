// Main JavaScript for NASA Exoplanet Detection System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading states to forms
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.classList.add('loading');
            }
        });
    });

    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    // Observe all cards and sections
    document.querySelectorAll('.card, .hero-section, .container').forEach(el => {
        observer.observe(el);
    });

    // Form validation
    const inputs = document.querySelectorAll('input[required]');
    inputs.forEach(input => {
        input.addEventListener('blur', validateInput);
        input.addEventListener('input', clearValidation);
    });

    function validateInput(e) {
        const input = e.target;
        const value = input.value.trim();
        
        if (!value) {
            showInputError(input, 'This field is required');
        } else if (input.type === 'number') {
            const num = parseFloat(value);
            const min = parseFloat(input.min);
            const max = parseFloat(input.max);
            
            if (isNaN(num)) {
                showInputError(input, 'Please enter a valid number');
            } else if (min && num < min) {
                showInputError(input, `Value must be at least ${min}`);
            } else if (max && num > max) {
                showInputError(input, `Value must be at most ${max}`);
            } else {
                clearInputError(input);
            }
        } else {
            clearInputError(input);
        }
    }

    function clearValidation(e) {
        clearInputError(e.target);
    }

    function showInputError(input, message) {
        clearInputError(input);
        input.classList.add('is-invalid');
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'invalid-feedback';
        errorDiv.textContent = message;
        input.parentNode.appendChild(errorDiv);
    }

    function clearInputError(input) {
        input.classList.remove('is-invalid');
        const errorDiv = input.parentNode.querySelector('.invalid-feedback');
        if (errorDiv) {
            errorDiv.remove();
        }
    }

    // Auto-save form data
    const formInputs = document.querySelectorAll('input, select, textarea');
    formInputs.forEach(input => {
        // Load saved data
        const savedValue = localStorage.getItem(`form_${input.name}`);
        if (savedValue && !input.value) {
            input.value = savedValue;
        }

        // Save data on change
        input.addEventListener('change', function() {
            localStorage.setItem(`form_${this.name}`, this.value);
        });
    });

    // Performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', function() {
            setTimeout(function() {
                const perfData = performance.getEntriesByType('navigation')[0];
                console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
            }, 0);
        });
    }

    // Error handling for fetch requests
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection:', event.reason);
        showNotification('An error occurred. Please try again.', 'error');
    });

    // Notification system
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 100px; right: 20px; z-index: 9999; min-width: 300px;';
        
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    // Make notification function globally available
    window.showNotification = showNotification;

    // Chart.js defaults
    if (typeof Chart !== 'undefined') {
        Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
        Chart.defaults.color = '#6c757d';
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit forms
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const form = e.target.closest('form');
            if (form) {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to close modals/alerts
        if (e.key === 'Escape') {
            const activeModal = document.querySelector('.modal.show');
            if (activeModal) {
                const modal = bootstrap.Modal.getInstance(activeModal);
                if (modal) modal.hide();
            }
            
            const activeAlert = document.querySelector('.alert.show');
            if (activeAlert) {
                const alert = bootstrap.Alert.getInstance(activeAlert);
                if (alert) alert.close();
            }
        }
    });

    // Responsive table handling
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
        if (table.scrollWidth > table.clientWidth) {
            table.parentNode.style.overflowX = 'auto';
        }
    });

    // Lazy loading for images
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });

        document.querySelectorAll('img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    }

    // Copy to clipboard functionality
    window.copyToClipboard = function(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                showNotification('Copied to clipboard!', 'success');
            }).catch(() => {
                fallbackCopyToClipboard(text);
            });
        } else {
            fallbackCopyToClipboard(text);
        }
    };

    function fallbackCopyToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            showNotification('Copied to clipboard!', 'success');
        } catch (err) {
            showNotification('Failed to copy to clipboard', 'error');
        }
        
        document.body.removeChild(textArea);
    }

    // Theme toggle (if needed)
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-theme');
            const isDark = document.body.classList.contains('dark-theme');
            localStorage.setItem('darkTheme', isDark);
        });

        // Load saved theme
        if (localStorage.getItem('darkTheme') === 'true') {
            document.body.classList.add('dark-theme');
        }
    }

    // Print functionality
    window.printPage = function() {
        window.print();
    };

    // Export functionality
    window.exportData = function(format = 'json') {
        // This would be implemented based on specific data export needs
        showNotification(`Exporting data as ${format.toUpperCase()}...`, 'info');
    };

    // Search functionality
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            const searchableElements = document.querySelectorAll('[data-searchable]');
            
            searchableElements.forEach(element => {
                const text = element.textContent.toLowerCase();
                if (text.includes(query)) {
                    element.style.display = '';
                } else {
                    element.style.display = 'none';
                }
            });
        });
    }

    // Initialize any additional components
    initializeComponents();
});

function initializeComponents() {
    // Initialize any additional JavaScript components here
    console.log('NASA Exoplanet Detection System initialized');
    
    // Initialize visualizer if we're on the visualizer page
    if (document.getElementById('planet-canvas')) {
        initializeVisualizer();
        // Start animation automatically
        startAnimation();
    }
    
    // Handle preset cards on visualizer page
    const presetCards = document.querySelectorAll('.preset-card');
    presetCards.forEach(card => {
        card.addEventListener('click', function() {
            const preset = this.dataset.preset;
            loadPreset(preset);
        });
    });
    
    // Handle visualizer form submission
    const visualizerForm = document.getElementById('visualizer-form');
    if (visualizerForm) {
        visualizerForm.addEventListener('submit', function(e) {
            e.preventDefault(); // Prevent form submission
            updateVisualizerFromForm();
            return false; // Ensure no refresh
        });
    }
    
    // Handle prediction form submission
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading indicator
            const loadingBar = document.getElementById('loading-bar');
            if (loadingBar) {
                loadingBar.style.display = 'block';
                const progressBar = document.getElementById('progress-bar');
                if (progressBar) {
                    progressBar.style.width = '100%';
                }
            }
            
            // Get form data and submit
            const formData = new FormData(this);
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                if (loadingBar) {
                    loadingBar.style.display = 'none';
                }
                
                if (data.success) {
                    // Update UI elements
                    const resultsSection = document.getElementById('results-section');
                    if (resultsSection) {
                        resultsSection.style.display = 'block';
                    }
                    
                    // Update classification text and confidence
                    const classificationText = document.getElementById('classification-text');
                    const confidenceBar = document.getElementById('confidence-bar');
                    const confidenceText = document.getElementById('confidence-text');
                    
                    if (classificationText) {
                        classificationText.textContent = data.classification;
                    }
                    
                    if (confidenceBar) {
                        confidenceBar.style.width = `${data.confidence}%`;
                        confidenceBar.className = 'progress-bar';
                        if (data.classification.includes('Confirmed')) {
                            confidenceBar.classList.add('bg-success');
                        } else if (data.classification.includes('False Positive')) {
                            confidenceBar.classList.add('bg-warning');
                        } else {
                            confidenceBar.classList.add('bg-danger');
                        }
                    }
                    
                    if (confidenceText) {
                        confidenceText.textContent = `${data.confidence}%`;
                    }
                    
                    // Do not show modal; keep results inline for better UX
                    // Ensure page remains scrollable
                    document.body.classList.remove('modal-open');
                    document.querySelectorAll('.modal-backdrop').forEach(el => el.remove());
                } else {
                    // Show error message
                    showNotification(data.error || 'An error occurred during prediction', 'error');
                }
            })
            .catch(error => {
                // Hide loading indicator and show error
                if (loadingBar) {
                    loadingBar.style.display = 'none';
                }
                console.error('Error:', error);
                showNotification('An error occurred while processing your request', 'error');
            });
        });
    }
}

// Visualizer functionality for separate visualizer page
function loadPreset(preset) {
    const presets = {
        earth: { period: 365, radius: 1.0, temp: 5778 },
        jupiter: { period: 4333, radius: 11.2, temp: 5778 },
        hotjupiter: { period: 3, radius: 10.0, temp: 6000 },
        superearth: { period: 100, radius: 2.5, temp: 5500 }
    };
    
    const data = presets[preset];
    if (data) {
        document.getElementById('vis-orbital-period').value = data.period;
        document.getElementById('vis-planet-radius').value = data.radius;
        document.getElementById('vis-stellar-temp').value = data.temp;
        updateVisualizerFromForm();
    }
}

function updateVisualizerFromForm() {
    const period = parseFloat(document.getElementById('vis-orbital-period').value) || 365;
    const radius = parseFloat(document.getElementById('vis-planet-radius').value) || 1.0;
    const temp = parseFloat(document.getElementById('vis-stellar-temp').value) || 5778;
    
    // Update system info
    document.getElementById('orbit-info').textContent = `${period} days`;
    document.getElementById('planet-info').textContent = `${radius} R⊕`;
    document.getElementById('star-info').textContent = `${temp}K star`;
    document.getElementById('distance-info').textContent = `${(period/365)**(2/3).toFixed(2)} AU`;
    
    // Update visualizer parameters
    const canvas = document.getElementById('planet-canvas');
    if (canvas) {
        // Update orbit size based on period
        const orbitRadius = 80 + (40 * Math.log10(period/365));
        
        // Update planet size based on radius
        const planetRadius = 4 + (2 * Math.log10(radius + 1));
        
        // Update star color based on temperature
        const starColor = getStarColor(temp);
        
        // Store parameters globally
        window.visualizerParams = {
            orbitRadius,
            planetRadius,
            starColor,
            period: period
        };
        
        // Reset animation with new parameters
        resetAnimation();
        if (isPlaying) {
            startAnimation();
        }
    }
}

function handlePredictionSubmission() {
    const form = document.getElementById('prediction-form');
    const submitBtn = form.querySelector('button[type="submit"]');
    const formData = new FormData(form);
    
    // Add loading state to button
    submitBtn.disabled = true;
    submitBtn.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
        Analyzing...
    `;
    
    // Show loading bar
    showLoadingBar();
    
    // Submit the form data
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingBar();
        
        if (data.success) {
            displayPredictionResults(data);
        } else {
            showNotification(data.error || 'An error occurred during prediction', 'error');
        }
    })
    .catch(error => {
        hideLoadingBar();
        console.error('Error:', error);
        showNotification('An error occurred while processing your request', 'error');
    });
}

function displayPredictionResults(data) {
    // Show results section
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.style.margin = '0 auto';
        resultsSection.style.maxWidth = '800px';
    }
    
    // Update classification status indicators
    const indicators = {
        confirmed: document.querySelector('.card.border-success'),
        potential: document.querySelector('.card.border-warning'),
        negative: document.querySelector('.card.border-danger')
    };
    
    // Reset all indicators
    Object.values(indicators).forEach(card => {
        if (card) {
            card.style.opacity = '0.5';
            card.style.transform = 'scale(0.95)';
        }
    });
    
    // Highlight active status
    let activeCard = null;
    if (data.classification.includes('Confirmed')) {
        activeCard = indicators.confirmed;
    } else if (data.classification.includes('False Positive')) {
        activeCard = indicators.potential;
    } else {
        activeCard = indicators.negative;
    }
    
    if (activeCard) {
        activeCard.style.opacity = '1';
        activeCard.style.transform = 'scale(1.05)';
        activeCard.style.boxShadow = '0 8px 24px rgba(0,113,220,0.2)';
    }

    // Update both modal and main page elements
    const elements = {
        modal: {
            classification: document.getElementById('modal-classification-text'),
            confidenceBar: document.getElementById('modal-confidence-bar'),
            confidenceText: document.getElementById('modal-confidence-text'),
            confidenceDescription: document.getElementById('modal-confidence-description'),
            header: document.getElementById('modal-header')
        },
        main: {
            classification: document.getElementById('classification-text'),
            confidenceBar: document.getElementById('confidence-bar'),
            confidenceText: document.getElementById('confidence-text'),
            confidenceDescription: document.getElementById('confidence-description')
        }
    };

    // Helper function to update elements
    const updateElements = (els, data) => {
        if (els.classification) els.classification.textContent = data.classification;
        if (els.confidenceBar) {
            els.confidenceBar.style.width = `${data.confidence}%`;
            els.confidenceBar.className = 'progress-bar';
            if (data.classification.includes('Confirmed')) {
                els.confidenceBar.classList.add('bg-success');
            } else if (data.classification.includes('False Positive')) {
                els.confidenceBar.classList.add('bg-warning');
            } else {
                els.confidenceBar.classList.add('bg-danger');
            }
        }
        if (els.confidenceText) els.confidenceText.textContent = `${data.confidence}%`;
        if (els.confidenceDescription) els.confidenceDescription.textContent = 
            `Confidence: ${data.confidence}% - ${data.classification}`;
    };

    // Update both modal and main page
    updateElements(elements.modal, data);
    updateElements(elements.main, data);

    // Update modal header style
    if (elements.modal.header) {
        elements.modal.header.className = 'modal-header';
        if (data.classification.includes('Confirmed')) {
            elements.modal.header.classList.add('bg-success', 'text-white');
        } else if (data.classification.includes('False Positive')) {
            elements.modal.header.classList.add('bg-warning', 'text-dark');
        } else {
            elements.modal.header.classList.add('bg-danger', 'text-white');
        }
    }

    // Skip modal display; show inline results only to avoid page freeze
    document.body.classList.remove('modal-open');
    document.querySelectorAll('.modal-backdrop').forEach(el => el.remove());
    
    showNotification('Classification completed successfully!', 'success');
}

// Visualizer functionality
let animationId;
let isPlaying = false;
let animationSpeed = 1;
let viewMode = 'top';
let planetAngle = 0;

function initializeVisualizer() {
    const canvas = document.getElementById('planet-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Set up controls
    const speedControl = document.getElementById('speed-control');
    const viewModeSelect = document.getElementById('view-mode');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    if (speedControl) {
        speedControl.addEventListener('input', function() {
            animationSpeed = parseFloat(this.value);
        });
    }
    
    if (viewModeSelect) {
        viewModeSelect.addEventListener('change', function() {
            viewMode = this.value;
            drawSystem();
        });
    }
    
    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', function() {
            if (isPlaying) {
                pauseAnimation();
            } else {
                startAnimation();
            }
        });
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            resetAnimation();
        });
    }
    
    // Initial draw
    drawSystem();
}

function startAnimation() {
    isPlaying = true;
    const playPauseBtn = document.getElementById('play-pause-btn');
    if (playPauseBtn) {
        playPauseBtn.innerHTML = '<i class="fas fa-pause"></i> Pause';
    }
    animate();
}

function pauseAnimation() {
    isPlaying = false;
    const playPauseBtn = document.getElementById('play-pause-btn');
    if (playPauseBtn) {
        playPauseBtn.innerHTML = '<i class="fas fa-play"></i> Play';
    }
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
}

function resetAnimation() {
    pauseAnimation();
    planetAngle = 0;
    drawSystem();
}

function animate() {
    if (!isPlaying) return;
    
    // Smooth animation based on parameters
    const params = window.visualizerParams || {
        period: 365,
        orbitRadius: 120
    };
    
    // Calculate angular velocity based on orbital period
    // Shorter period = faster rotation
    const baseSpeed = 0.02;
    const periodFactor = Math.sqrt(365 / params.period);
    const adjustedSpeed = baseSpeed * periodFactor * animationSpeed;
    
    planetAngle += adjustedSpeed;
    
    // Add slight oscillation to planet's vertical position for 3D effect
    const oscillation = Math.sin(planetAngle * 2) * 5;
    
    // Update system with oscillation
    drawSystem(oscillation);
    animationId = requestAnimationFrame(animate);
}

function getStarColor(temp) {
    // Convert temperature to RGB using approximation
    if (temp < 3500) {
        return '#ff4500'; // Red/Orange for cool stars
    } else if (temp < 5000) {
        return '#ffa500'; // Orange/Yellow for K-type stars
    } else if (temp < 6000) {
        return '#ffff00'; // Yellow for G-type stars (like our Sun)
    } else if (temp < 7500) {
        return '#f8f8ff'; // White for F-type stars
    } else {
        return '#caf0f8'; // Blue-white for hot stars
    }
}

function drawSystem() {
    const canvas = document.getElementById('planet-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // Get visualization parameters
    const params = window.visualizerParams || {
        orbitRadius: 120,
        planetRadius: 8,
        starColor: '#ffff00',
        period: 365
    };
    
    // Clear canvas
    ctx.fillStyle = '#0c0c0c';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw stars background
    drawStars(ctx, canvas.width, canvas.height);
    
    // Draw star (center)
    drawStar(ctx, centerX, centerY, 20, params.starColor);
    
    // Draw orbit
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, params.orbitRadius, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Draw planet
    const planetX = centerX + Math.cos(planetAngle) * params.orbitRadius;
    const planetY = centerY + Math.sin(planetAngle) * params.orbitRadius;
    drawPlanet(ctx, planetX, planetY, params.planetRadius);
    
    // Draw planet trail
    drawPlanetTrail(ctx, centerX, centerY, orbitRadius);
}

function drawStars(ctx, width, height) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    for (let i = 0; i < 50; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        const size = Math.random() * 2;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, 2 * Math.PI);
        ctx.fill();
    }
}

function drawStar(ctx, x, y, radius, color = '#FFD700') {
    // Convert hex color to RGB for glow effect
    let r, g, b;
    if (color.startsWith('#')) {
        const hex = color.substring(1);
        r = parseInt(hex.substring(0, 2), 16);
        g = parseInt(hex.substring(2, 4), 16);
        b = parseInt(hex.substring(4, 6), 16);
    }
    
    // Star glow
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 2);
    gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.8)`);
    gradient.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, 0.4)`);
    gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius * 2, 0, 2 * Math.PI);
    ctx.fill();
    
    // Star core
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
}

function drawPlanet(ctx, x, y, radius, oscillation = 0) {
    // Apply vertical oscillation for 3D effect
    const adjustedY = y + oscillation;
    
    // Planet shadow/atmosphere
    const atmosphereGradient = ctx.createRadialGradient(x, adjustedY, radius, x, adjustedY, radius * 2);
    atmosphereGradient.addColorStop(0, 'rgba(100, 150, 255, 0.3)');
    atmosphereGradient.addColorStop(0.5, 'rgba(50, 100, 200, 0.1)');
    atmosphereGradient.addColorStop(1, 'rgba(0, 50, 150, 0)');
    
    ctx.fillStyle = atmosphereGradient;
    ctx.beginPath();
    ctx.arc(x, adjustedY, radius * 2, 0, 2 * Math.PI);
    ctx.fill();
    
    // Planet glow
    const glowGradient = ctx.createRadialGradient(x, adjustedY, radius * 0.8, x, adjustedY, radius * 1.5);
    glowGradient.addColorStop(0, 'rgba(100, 150, 255, 0.8)');
    glowGradient.addColorStop(0.7, 'rgba(50, 100, 200, 0.4)');
    glowGradient.addColorStop(1, 'rgba(0, 50, 150, 0)');
    
    ctx.fillStyle = glowGradient;
    ctx.beginPath();
    ctx.arc(x, adjustedY, radius * 1.5, 0, 2 * Math.PI);
    ctx.fill();
    
    // Planet base
    const planetGradient = ctx.createRadialGradient(x - radius * 0.3, adjustedY - radius * 0.3, 0, x, adjustedY, radius);
    planetGradient.addColorStop(0, '#6AB7FF');
    planetGradient.addColorStop(0.5, '#4A90E2');
    planetGradient.addColorStop(1, '#2171CD');
    
    ctx.fillStyle = planetGradient;
    ctx.beginPath();
    ctx.arc(x, adjustedY, radius, 0, 2 * Math.PI);
    ctx.fill();
    
    // Planet highlight/specular
    const highlightGradient = ctx.createRadialGradient(
        x - radius * 0.3, 
        adjustedY - radius * 0.3, 
        0,
        x - radius * 0.3,
        adjustedY - radius * 0.3,
        radius * 0.6
    );
    highlightGradient.addColorStop(0, 'rgba(255, 255, 255, 0.4)');
    highlightGradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.1)');
    highlightGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    
    ctx.fillStyle = highlightGradient;
    ctx.beginPath();
    ctx.arc(x - radius * 0.3, adjustedY - radius * 0.3, radius * 0.6, 0, 2 * Math.PI);
    ctx.fill();
}

function drawPlanetTrail(ctx, centerX, centerY, orbitRadius) {
    ctx.strokeStyle = 'rgba(100, 150, 255, 0.2)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < 20; i++) {
        const angle = planetAngle - (i * 0.1);
        const x = centerX + Math.cos(angle) * orbitRadius;
        const y = centerY + Math.sin(angle) * orbitRadius;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
}

// Make functions globally available
window.initializeVisualizer = initializeVisualizer;
window.updateVisualizer = function(params) {
    // Update visualizer based on parameters
    drawSystem();
};

// Utility functions
const utils = {
    // Format numbers with commas
    formatNumber: function(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    },

    // Format percentages
    formatPercentage: function(num, decimals = 1) {
        return (num * 100).toFixed(decimals) + '%';
    },

    // Debounce function
    debounce: function(func, wait, immediate) {
        let timeout;
        return function executedFunction() {
            const context = this;
            const args = arguments;
            const later = function() {
                timeout = null;
                if (!immediate) func.apply(context, args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func.apply(context, args);
        };
    },

    // Throttle function
    throttle: function(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
};

// Make utils globally available
window.utils = utils;
