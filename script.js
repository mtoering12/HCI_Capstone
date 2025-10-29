const map = L.map('map').setView([37.7749, -122.4194], 5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Blue Ridge Trail
L.circleMarker([38.0293, -78.4767], {
  color: 'green',
  radius: 10
}).addTo(map).bindPopup("Blue Ridge Trail");

// Cascades Falls Trail
L.circleMarker([37.3535, -80.5996], {
  color: 'blue',
  radius: 10
}).addTo(map).bindPopup("Cascades Falls Trail");

// McAfee Knob Trail
L.circleMarker([37.3806, -80.0890], {
  color: 'red',
  radius: 10
}).addTo(map).bindPopup("McAfee Knob Trail");

// Search filter
document.getElementById('searchInput').addEventListener('input', function (e) {
  const query = e.target.value.toLowerCase();
  const cards = document.querySelectorAll('.trail-card');

  cards.forEach(card => {
    const text = card.textContent.toLowerCase();
    card.style.display = text.includes(query) ? 'block' : 'none';
  });
});
