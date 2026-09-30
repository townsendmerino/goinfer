// Download page: mark the rows for the visitor's platform. The page is complete without this.
(function () {
  var ua = navigator.userAgent || "", pl = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "";
  var os = /Mac/i.test(pl + ua) ? "darwin" : /Win/i.test(pl + ua) ? "windows" : /Linux|X11|CrOS/i.test(pl + ua) ? "linux" : "";
  if (!os) return;
  // The browser cannot tell Apple silicon from Intel reliably, so it marks every arch of the OS rather than guess.
  document.querySelectorAll("tr[data-platform^='" + os + "-']").forEach(function (tr) {
    tr.classList.add("mine");
    var h = tr.querySelector(".here");
    if (h) h.hidden = false;
  });
})();
