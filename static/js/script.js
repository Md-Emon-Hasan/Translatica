function translateText() {
      const input = document.getElementById("inputText").value.trim();
      const output = document.getElementById("output");
      if (!input) {
        output.innerHTML = "<span class='text-warning'>⚠️ Please enter some text!</span>";
        return;
      }

      output.innerHTML = "<em>⏳ Translating...</em>";

      fetch("/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      })
        .then((res) => res.json())
        .then((data) => {
          if (data.translation) {
            output.innerHTML = `<strong>✅ Translation:</strong><br>${data.translation}`;
          } else {
            output.innerHTML = `<span class="text-danger">❌ Error: ${data.error}</span>`;
          }
        })
        .catch((err) => {
          output.innerHTML = "<span class='text-danger'>⚠️ Failed to translate.</span>";
        });
    }