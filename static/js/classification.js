// Handle exoplanet classification form and results
document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('prediction-form');
    const resultsSection = document.getElementById('results-section');
    const loadingBar = document.getElementById('loading-bar');
    
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading bar
            loadingBar.style.display = 'block';
            animateProgress();
            
            // Get form data
            const formData = new FormData(this);
            
            // Make prediction request
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showResults(data);
                } else {
                    showError(data.error);
                }
            })
            .catch(error => {
                showError('An unexpected error occurred. Please try again.');
                console.error('Error:', error);
            })
            .finally(() => {
                loadingBar.style.display = 'none';
            });
        });
    }

    function showResults(data) {
        // Show results section
        resultsSection.style.display = 'block';
        
        // Update classification text and confidence
        const classificationText = document.getElementById('classification-text');
        const confidenceBar = document.getElementById('confidence-bar');
        const confidenceText = document.getElementById('confidence-text');
        const confidenceDescription = document.getElementById('confidence-description');
        
        classificationText.textContent = data.classification;
        confidenceText.textContent = data.confidence + '%';
        confidenceBar.style.width = data.confidence + '%';
        
        // Reset all classification cards to default state
        ['confirmed-card','potential-card','not-exoplanet-card'].forEach(id => {
            const c = document.getElementById(id);
            if (c) c.classList.remove('selected-card','selected-success','selected-warning','selected-danger');
        });
        
        // Highlight the appropriate card based on classification
        let cardId, selectedClass, barClass;
        switch(data.classification) {
            case 'Confirmed Exoplanet':
                cardId = 'confirmed-card';
                selectedClass = 'selected-card selected-success';
                barClass = 'bg-success';
                break;
            case 'Potential False Positive':
                cardId = 'potential-card';
                selectedClass = 'selected-card selected-warning';
                barClass = 'bg-warning';
                break;
            case 'Not an Exoplanet':
                cardId = 'not-exoplanet-card';
                selectedClass = 'selected-card selected-danger';
                barClass = 'bg-danger';
                break;
        }
        
        // Apply highlighting to selected card
        const selectedCard = document.getElementById(cardId);
        if (selectedCard) {
            selectedCard.classList.add(...selectedClass.split(' '));
        }
        // Update progress bar color
        confidenceBar.className = `progress-bar ${barClass}`;
        
        // Update descriptive text under the bar
        let desc;
        if (data.classification === 'Confirmed Exoplanet') {
            desc = 'High confidence prediction that the observed signal is a real exoplanet.';
        } else if (data.classification === 'Potential False Positive') {
            desc = 'Intermediate confidence; further checks recommended to rule out false positives.';
        } else {
            desc = 'Low confidence for an exoplanet signal; likely a non-planetary source or noise.';
        }
        if (confidenceDescription) confidenceDescription.textContent = desc;
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        // Hide results section
        resultsSection.style.display = 'none';
        
        // Show error message
        alert(message);
    }

    function animateProgress() {
        const progressBar = document.getElementById('progress-bar');
        const loadingText = document.getElementById('loading-text');
        let progress = 0;
        
        const interval = setInterval(() => {
            if (progress >= 90) {
                clearInterval(interval);
                return;
            }
            progress += Math.random() * 30;
            if (progress > 90) progress = 90;
            
            progressBar.style.width = progress + '%';
            loadingText.textContent = 'Analyzing... ' + Math.round(progress) + '%';
        }, 500);
    }
});