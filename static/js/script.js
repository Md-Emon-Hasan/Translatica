// Translatica - Translation Script

function translateText() {
  const input = document.getElementById("inputText").value.trim();
  const output = document.getElementById("output");
  const btn = document.querySelector(".translate-btn");
  
  // Validate input
  if (!input) {
    output.style.display = "block";
    output.innerHTML = `
      <div class="error-message">
        <i class="fas fa-exclamation-circle"></i>
        <span>Please enter some text to translate!</span>
      </div>
    `;
    return;
  }

  // Show loading state
  output.style.display = "block";
  output.innerHTML = `
    <div class="loading-state">
      <div class="spinner"></div>
      <span>Translating your text...</span>
    </div>
  `;
  
  // Disable button during translation
  btn.disabled = true;
  btn.classList.add("loading");

  // Make API call
  fetch("/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: input }),
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.translation) {
        output.innerHTML = `
          <div class="success-result">
            <div class="result-header">
              <i class="fas fa-check-circle"></i>
              <span>Translation Complete</span>
            </div>
            <div class="result-text">${escapeHtml(data.translation)}</div>
            <button class="copy-btn" onclick="copyToClipboard(\`${escapeHtml(data.translation)}\`)">
              <i class="fas fa-copy"></i> Copy
            </button>
          </div>
        `;
      } else {
        output.innerHTML = `
          <div class="error-message">
            <i class="fas fa-times-circle"></i>
            <span>Error: ${escapeHtml(data.error || 'Unknown error occurred')}</span>
          </div>
        `;
      }
    })
    .catch((err) => {
      console.error('Translation error:', err);
      output.innerHTML = `
        <div class="error-message">
          <i class="fas fa-exclamation-triangle"></i>
          <span>Failed to connect to translation service. Please try again.</span>
        </div>
      `;
    })
    .finally(() => {
      btn.disabled = false;
      btn.classList.remove("loading");
    });
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Copy to clipboard function
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    const copyBtn = document.querySelector(".copy-btn");
    const originalText = copyBtn.innerHTML;
    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    copyBtn.classList.add("copied");
    setTimeout(() => {
      copyBtn.innerHTML = originalText;
      copyBtn.classList.remove("copied");
    }, 2000);
  }).catch(err => {
    console.error('Failed to copy:', err);
    alert('Failed to copy to clipboard');
  });
}

// Character counter
document.addEventListener("DOMContentLoaded", () => {
  const textarea = document.getElementById("inputText");
  const charCount = document.getElementById("charCount");
  
  if (textarea && charCount) {
    textarea.addEventListener("input", () => {
      const len = textarea.value.length;
      charCount.textContent = `${len} / 500`;
      if (len > 500) {
        charCount.classList.add("exceeded");
      } else {
        charCount.classList.remove("exceeded");
      }
    });
  }

  // Create floating particles
  createParticles();
});

// Floating particles animation
function createParticles() {
  const container = document.getElementById("particles");
  if (!container) return;
  
  for (let i = 0; i < 50; i++) {
    const particle = document.createElement("div");
    particle.className = "particle";
    particle.style.left = Math.random() * 100 + "%";
    particle.style.animationDelay = Math.random() * 20 + "s";
    particle.style.animationDuration = (15 + Math.random() * 20) + "s";
    container.appendChild(particle);
  }
}

// Keyboard shortcut: Ctrl/Cmd + Enter to translate
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    translateText();
  }
});