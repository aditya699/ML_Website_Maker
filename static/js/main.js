// Get form and result elements
const form = document.getElementById('predictionForm');
const result = document.getElementById('result');
const priceDisplay = document.getElementById('predictedPrice');

// Handle form submission
form.onsubmit = function(e) {
    e.preventDefault();
    
    // Create simple object with form data
    const data = {
        square_feet: form.square_feet.value,
        bedrooms: form.bedrooms.value,
        bathrooms: form.bathrooms.value,
        age: form.age.value
    };

    // Send prediction request
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        // Show result
        result.style.display = 'block';
        priceDisplay.textContent = '$' + Number(data.predicted_price).toLocaleString();
    })
    .catch(error => {
        alert('Error making prediction. Please try again.');
    });
};