const map = L.map('map').setView([37.7749, -122.4194], 5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Blue Ridge Trail
L.circleMarker([38.0293, -78.4767], {
  color: 'green',
  radius: 10
}).addTo(map).bindPopup(`
  <strong>Blue Ridge Trail</strong><br/>
  <em>Peaceful and scenic route through Virginia’s Blue Ridge Mountains.</em><br/>
  <ul>
    <li>🦋 West Virginia White butterflies spotted</li>
    <li>🐦 Migratory warblers active</li>
    <li>🐾 Mountain lion sighting near Skyline Drive</li>
  </ul>
`);

// Cascades Falls Trail
L.circleMarker([37.3535, -80.5996], {
  color: 'blue',
  radius: 10
}).addTo(map).bindPopup(`
  <strong>Cascades Falls Trail</strong><br/>
  <em>Moderate hike to a 66-foot waterfall, known for serenity and rhododendron tunnels.</em><br/>
  <ul>
    <li>🐕 Dog-friendly with light crowds</li>
    <li>🌸 Rhododendron blooms along creek</li>
    <li>🌧️ Muddy conditions recently cleared</li>
  </ul>
`);

// McAfee Knob Trail
L.circleMarker([37.3806, -80.0890], {
  color: 'red',
  radius: 10
}).addTo(map).bindPopup(`
  <strong>McAfee Knob Trail</strong><br/>
  <em>Iconic Appalachian hike with panoramic views and emotional exhilaration.</em><br/>
  <ul>
    <li>🐿️ Wildlife: deer, chipmunks, hawks</li>
    <li>🌄 Popular for sunrise hikes</li>
    <li>🛑 Weekend congestion reported</li>
  </ul>
`);

// Search filter
document.getElementById('searchInput').addEventListener('input', function (e) {
  const query = e.target.value.toLowerCase();
  const cards = document.querySelectorAll('.trail-card');

  cards.forEach(card => {
    const text = card.textContent.toLowerCase();
    card.style.display = text.includes(query) ? 'block' : 'none';
  });
});
