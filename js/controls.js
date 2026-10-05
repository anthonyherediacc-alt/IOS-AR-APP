import { WOUNDS, SETTINGS, TRACKING } from "./config.js";

// Side panel: live position/size offsets, steadiness, a user-supplied picture, and reset to defaults.
// Edits the active wound entry in place; the renderer reads it every frame.
const DEFAULTS = { woundId: SETTINGS.woundId, wounds: WOUNDS.map((w) => ({ ...w })), steadiness: TRACKING.steadiness };
const FIELDS = { ctlX: "xOffset", ctlY: "yOffset", ctlScale: "scale" };
let customCount = 0;

export function initControls(currentWound, showError) {
  const $ = (id) => document.getElementById(id);
  const toggle = $("panelToggle"), panel = $("panel"), file = $("ctlFile");

  toggle.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    toggle.setAttribute("aria-expanded", String(!panel.hidden));
  });

  const sync = () => {
    for (const [id, key] of Object.entries(FIELDS)) $(id).value = currentWound()[key];
    $("ctlSteady").value = TRACKING.steadiness;
  };
  $("ctlSteady").addEventListener("input", (e) => { TRACKING.steadiness = Number(e.target.value); });
  for (const [id, key] of Object.entries(FIELDS)) {
    $(id).addEventListener("input", (e) => { currentWound()[key] = Number(e.target.value); });
  }

  file.addEventListener("change", () => {
    const f = file.files[0];
    if (!f) return;
    const src = URL.createObjectURL(f);
    const img = new Image();
    img.onload = () => {
      const base = currentWound();
      const wound = {
        id: `custom-${++customCount}`, name: "My picture", src, anchor: "handCenter",
        scale: base.scale, rotationOffset: 0, xOffset: base.xOffset, yOffset: base.yOffset, opacity: 1, roughness: 0.8,
      };
      removeCustom();
      WOUNDS.push(wound);
      SETTINGS.woundId = wound.id;
      sync();
    };
    img.onerror = () => { URL.revokeObjectURL(src); showError("Couldn't open that picture. Try a PNG or JPEG."); };
    img.src = src;
  });

  $("ctlReset").addEventListener("click", () => {
    removeCustom();
    DEFAULTS.wounds.forEach((d, i) => Object.assign(WOUNDS[i], d));
    SETTINGS.woundId = DEFAULTS.woundId;
    TRACKING.steadiness = DEFAULTS.steadiness;
    file.value = "";
    sync();
  });

  sync();
}

function removeCustom() {
  for (let i = WOUNDS.length - 1; i >= DEFAULTS.wounds.length; i--) {
    URL.revokeObjectURL(WOUNDS[i].src);
    WOUNDS.splice(i, 1);
  }
}
