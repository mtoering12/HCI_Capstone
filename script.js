const map = L.map('map').setView([37.7749, -122.4194], 5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

const trailPin = L.circleMarker([38.0293, -78.4767], {
  color: 'green',
  radius: 10
}).addTo(map).bindPopup("Blue Ridge Trail");

document.getElementById('searchInput').addEventListener('input', function (e) {
  const query = e.target.value.toLowerCase();
  const cards = document.querySelectorAll('.trail-card');

  cards.forEach(card => {
    const text = card.textContent.toLowerCase();
    card.style.display = text.includes(query) ? 'block' : 'none';
  });
});
