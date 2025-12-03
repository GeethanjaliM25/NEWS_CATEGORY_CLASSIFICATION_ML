// Function to send text to Flask backend and get prediction
async function predictCategory() {
  const text = document.getElementById("newsInput").value.trim();
  const resultDiv = document.getElementById("result");

  // Clear previous result
  resultDiv.innerText = "";

  if (!text) {
    resultDiv.innerText = "⚠️ Please enter some text before predicting.";
    return;
  }

  resultDiv.innerText = "⏳ Predicting category... please wait.";

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      throw new Error("Backend error or wrong URL");
    }

    const data = await response.json();

    if (data.error) {
      resultDiv.innerText = `⚠️ ${data.error}`;
    } else {
      resultDiv.innerHTML = `
        ✅ <strong>Predicted Category:</strong> ${data.category}<br>
        🏷️ <strong>Class ID:</strong> ${data.predicted_class}
      `;
    }
  } catch (err) {
    resultDiv.innerText = "❌ Unable to connect to backend. Make sure Flask is running at http://127.0.0.1:5000";
    console.error(err);
  }
}

// Load example news text for testing
function loadExample(id) {
  const examples = {
    1: "UN Secretary addresses global warming and peace talks at the climate summit.",
    2: "India defeats Australia in a thrilling T20 World Cup final match.",
    3: "Stock market soars as tech companies post record quarterly profits.",
    4: "NASA launches a new satellite to explore the surface of Mars."
  };

  document.getElementById("newsInput").value = examples[id];
  document.getElementById("result").innerText = "";
}
