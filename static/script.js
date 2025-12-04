document.addEventListener("DOMContentLoaded", function() {
  // ---------------- Map initialization ----------------
  const map = L.map('map').setView([37.7749, -122.4194], 5);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);

  // Trail markers
  L.circleMarker([38.0293, -78.4767], { color: 'green', radius: 10 }).addTo(map)
    .bindPopup("<strong>Blue Ridge Trail</strong>...");

  L.circleMarker([37.3535, -80.5996], { color: 'blue', radius: 10 }).addTo(map)
    .bindPopup("<strong>Cascades Falls Trail</strong>...");

  L.circleMarker([37.3806, -80.0890], { color: 'red', radius: 10 }).addTo(map)
    .bindPopup("<strong>McAfee Knob Trail</strong>...");

  // ---------------- Search filter ----------------
  document.getElementById('searchInput').addEventListener('input', function (e) {
    const query = e.target.value.toLowerCase();
    const cards = document.querySelectorAll('.trail-card');
    cards.forEach(card => {
      const text = card.textContent.toLowerCase();
      card.style.display = text.includes(query) ? 'block' : 'none';
    });
  });

  // ---------------- Chat template system ----------------
  const FILL_IN_TEMPLATES = [
    { title: "I want to visit...", template: "I want to visit {location}. What can you tell me about trails or posts mentioning it?", placeholders: ["location"] },
    { title: "I am... and I want...", template: "I am {self_desc}, and I want {goal}. Which trails or posts might fit?", placeholders: ["self_desc", "goal"] },
    { title: "What are people saying this season about...", template: "What are people saying this {season_topic}? Summarize key emotions or trail issues.", placeholders: ["season_topic"] },
    { title: "Tell me about trails that are...", template: "Tell me about trails that are {adjective}.", placeholders: ["adjective"] }
  ];

  const templateSelect = document.getElementById("templateSelect");
  const placeholdersContainer = document.getElementById("placeholdersContainer");

  // Populate template dropdown
  FILL_IN_TEMPLATES.forEach((tpl, idx) => {
    const option = document.createElement("option");
    option.value = idx;
    option.textContent = tpl.title;
    templateSelect.appendChild(option);
  });

  // Show input fields when template is selected
  templateSelect.addEventListener("change", () => {
    placeholdersContainer.innerHTML = "";
    const idx = templateSelect.value;
    if (idx === "") return;

    const template = FILL_IN_TEMPLATES[idx];
    template.placeholders.forEach(ph => {
      const label = document.createElement("label");
      label.textContent = ph + ": ";
      const input = document.createElement("input");
      input.type = "text";
      input.id = "ph_" + ph;
      input.placeholder = ph;
      label.appendChild(input);
      placeholdersContainer.appendChild(label);
      placeholdersContainer.appendChild(document.createElement("br"));
    });
  });

  // Send button
  document.getElementById("sendBtn").addEventListener("click", async () => {
    const idx = templateSelect.value;
    if (idx === "") {
      alert("Please select a template!");
      return;
    }

    const template = FILL_IN_TEMPLATES[idx];
    const placeholders = {};
    template.placeholders.forEach(ph => {
      placeholders[ph] = document.getElementById("ph_" + ph).value.trim();
    });

    const query = template.template.replace(/\{(\w+)\}/g, (_, key) => placeholders[key] || "");

    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, template_index: idx, placeholders })
    });

    const data = await res.json();
    document.getElementById("responseBox").textContent = data.response || data.error;
  });
});
