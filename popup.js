document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("increaseFont").addEventListener("click", function () {
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            chrome.scripting.executeScript({
                target: { tabId: tabs[0].id },
                func: () => {
                    document.querySelectorAll("*").forEach(el => {
                        let currentSize = window.getComputedStyle(el).fontSize;
                        el.style.fontSize = (parseFloat(currentSize) + 2) + "px";
                    });
                }
            });
        });
    });

    document.getElementById("decreaseFont").addEventListener("click", function () {
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            chrome.scripting.executeScript({
                target: { tabId: tabs[0].id },
                func: () => {
                    document.querySelectorAll("*").forEach(el => {
                        let currentSize = window.getComputedStyle(el).fontSize;
                        el.style.fontSize = (parseFloat(currentSize) - 2) + "px";
                    });
                }
            });
        });
    });

    document.getElementById("enableTTS").addEventListener("click", function () {
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            chrome.scripting.executeScript({
                target: { tabId: tabs[0].id },
                func: () => {
                    speechSynthesis.cancel(); // Stop any ongoing speech
                    let selectedText = window.getSelection().toString().trim();
                    if (selectedText.length === 0) {
                        alert("Please select text to read.");
                        return;
                    }
                    let utterance = new SpeechSynthesisUtterance(selectedText);
                    speechSynthesis.speak(utterance);
                }
            });
        });
    });

    document.getElementById("stopTTS").addEventListener("click", function () {
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            chrome.scripting.executeScript({
                target: { tabId: tabs[0].id },
                func: () => {
                    speechSynthesis.cancel(); // Stop speech immediately
                }
            });
        });
    });
});
