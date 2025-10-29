const map = L.map('map').setView([37.7749, -122.4194], 5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

const trailPin = L.circleMarker([38.0293, -78.4767], {
  color: 'green',
  radius: 10
}).addTo(map).bindPopup("Blue Ridge Trail").on('click', () => {
  document.getElementById('trail-modal').classList.remove('hidden');
});

function closeModal() {
  document.getElementById('trail-modal').classList.add('hidden');
}
