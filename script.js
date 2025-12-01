async function verifyNews() {
    const news = document.getElementById("newsInput").value.trim();
    const resultBox = document.getElementById("result");
    const confidenceBox = document.getElementById("confidence"); 
    const accuracyBox = document.getElementById("accuracy");
    resultBox.innerText = "";
    confidenceBox.innerText = ""; 
    accuracyBox.innerText = "";

    if (!news) {
        resultBox.innerText = "Please enter a news headline first.";
        resultBox.style.color = "#d90429";
        return;
    }

    resultBox.innerText = "Verifying...";
    resultBox.style.color = "#555";

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ news })
    });

    const data = await response.json();

    if (data.result === "Real News") {
        resultBox.innerText = " VERIFIED: REAL NEWS";
        resultBox.className = "real";
    } else {
        resultBox.innerText = " ALERT: FAKE NEWS DETECTED";
        resultBox.className = "fake";
    }

    confidenceBox.innerText = "Prediction Confidence: " + data.confidence; 
    accuracyBox.innerText = "Model Accuracy: " + data.accuracy; 
}
