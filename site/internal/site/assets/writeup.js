// Writeup pages: the mobile "what's different" list navigates on change.
(function () {
  var sel = document.getElementById("railsel");
  if (sel) sel.addEventListener("change", function () { location.href = sel.value; });
})();
