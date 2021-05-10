
 var leafletPolycolor = require('leaflet-polycolor');
 leafletPolycolor.default(L);

var map = L.map('map', {
    center: [13.755058571018745, 100.47227612726593],
    zoom: 30
});

L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png?{foo}', {foo: 'bar'}).addTo(map);

console.log(latlng1);
console.log(color1);
console.log(label_code);
console.log(label_pro);
//console.log(latLngs);
var polyline;


var arrayc;
for (i = 0; i < latlng1.length; i++) {
  arrayc=latlng1[i];
  colorse=color1[i];
  poly=L.polycolor(arrayc, {
    colors: colorse,
    weight: 5
  })
  
  poly.addTo(map);
  console.log(poly)

}



function zoomTo() {
  
  var r_code = document.getElementById('r_code').value;
  index_find=this.label_code.indexOf(r_code);
  console.log(index_find)

  var marker = L.marker([this.latlng1[index_find][0][0],this.latlng1[index_find][0][1]]).bindPopup('<center>'+label_code[index_find]+'<br><a href="http://192.168.1.15:5000/dashboard#roadcode='+label_code[index_find]+'&province='+label_pro[index_find]+'"> See more</a>').addTo(map);
  
  map.panTo(new L.LatLng(this.latlng1[index_find][0][0],this.latlng1[index_find][0][1]));

}   
var legend = L.control({ position: "bottomleft" });


legend.onAdd = function(map) {
  var div = L.DomUtil.create("div", "legend");
  div.innerHTML += "<h4>ช่วงค่า IRI</h4>";
  div.innerHTML += '<i style="background: green"></i><span>0.0-2.0</span><br>';
  div.innerHTML += '<i style="background: yellow"></i><span>2.0-3.5</span><br>';
  div.innerHTML += '<i style="background: orange"></i><span>3.5-5.0</span><br>';
  div.innerHTML += '<i style="background: red"></i><span>5.0 ขึ้นไป</span><br>';
  
  

  return div;
};

legend.addTo(map);





