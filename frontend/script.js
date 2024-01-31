async function analyzeReview() {
    let reviewText = document.getElementById('review-text').value;
    try {
        let response = await fetch('http://127.0.0.1:8000/analyze_review/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review_text: reviewText })
        });
        if (response.ok) {
            let result = await response.json();
            document.getElementById('result').innerText = result.dominant_topic;
        } else {
            document.getElementById('result').innerText = 'Error: ' + response.statusText;
        }
    } catch (error) {
        document.getElementById('result').innerText = 'Error: ' + error.message;
    }
}
