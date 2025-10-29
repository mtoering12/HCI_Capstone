const map = L.map('map').setView([37.7749, -122.4194], 5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Original Blue Ridge Trail pin
const blueRidgePin = L.circleMarker([38.0293, -78.4767], {
  color: 'green',
  radius: 10
}).addTo(map).bindPopup("Blue Ridge Trail");

// New: Cascades Falls Trail pin
const cascadesPin = L.circleMarker([37.3535, -80.5996], {
  color: 'blue',
  radius: 10
}).addTo(map).bindPopup("Cascades Falls Trail");

// New: McAfee Knob Trail pin
const mcafeePin = L.circleMarker([37.3806, -80.0890], {
  color: 'red',
  radius: 10
}).addTo(map).bindPopup("McAfee Knob Trail");

// Search filter logic
document.getElementById('searchInput').addEventListener('input', function (e) {
  const query = e.target.value.toLowerCase();
  const cards = document.querySelectorAll('.trail-card');

  cards.forEach(card => {
    const text = card.textContent.toLowerCase();
    card.style.display = text.includes(query) ? 'block' : 'none';
  });
});
