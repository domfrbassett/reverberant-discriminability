# Lists, and switches between run setups, the Pachyderm sources and receivers in the BRAS CR4 model.
# Run in Rhino with the CR4 model open and the Pachyderm plug-in loaded; choose a mode at the prompt.
#   Print : list every Pachyderm source and receiver with its coordinates and BRAS name
#   RunA1 : source LS1 only; receivers MP1-MP5 and the seat grid (room map)
#   RunA2 : source LS1 only; receivers MP1-MP5 only (order check, measured omni RIRs)
#   RunB  : source LS7 only; receiver MP1 (= MP6, the BRIR seat) only
import rhinoscriptsyntax as rs

TOL = 0.05

SOURCES = {
    "LS1": (0.00, 4.50, 1.38),
    "LS7": (-1.50, 0.00, 1.50),
}

MEASUREMENT_POINTS = {
    "MP1": (8.50, 0.00, 1.09),
    "MP2": (3.33, -7.95, 0.57),
    "MP3": (9.33, -6.96, 1.18),
    "MP4": (5.91, 6.34, 0.83),
    "MP5": (11.83, 8.43, 1.43),
}

SEAT_GRID = [
    (3.00, -12.00, 3.62),
    (3.00, -10.00, 0.53),
    (3.00, -8.00, 0.53),
    (3.00, -6.00, 0.53),
    (3.00, 6.00, 0.53),
    (3.00, 8.00, 0.53),
    (3.00, 10.00, 0.53),
    (3.00, 12.00, 3.62),
    (5.00, -12.00, 4.04),
    (5.00, -10.00, 0.74),
    (5.00, -8.00, 0.74),
    (5.00, -6.00, 0.74),
    (5.00, -4.00, 0.74),
    (5.00, -2.00, 0.74),
    (5.00, 2.00, 0.74),
    (5.00, 4.00, 0.74),
    (5.00, 6.00, 0.74),
    (5.00, 8.00, 0.74),
    (5.00, 10.00, 0.74),
    (5.00, 12.00, 4.04),
    (7.00, -12.00, 4.47),
    (7.00, -10.00, 0.94),
    (7.00, -8.00, 0.94),
    (7.00, -6.00, 0.94),
    (7.00, -4.00, 0.94),
    (7.00, -2.00, 0.94),
    (7.00, 2.00, 0.94),
    (7.00, 4.00, 0.94),
    (7.00, 6.00, 0.94),
    (7.00, 8.00, 0.94),
    (7.00, 10.00, 0.94),
    (7.00, 12.00, 4.47),
    (9.00, -12.00, 4.89),
    (9.00, -10.00, 1.14),
    (9.00, -8.00, 1.14),
    (9.00, -6.00, 1.14),
    (9.00, -4.00, 1.14),
    (9.00, -2.00, 1.14),
    (9.00, 2.00, 1.14),
    (9.00, 4.00, 1.14),
    (9.00, 6.00, 1.14),
    (9.00, 8.00, 1.14),
    (9.00, 10.00, 1.14),
    (9.00, 12.00, 4.89),
    (11.00, -12.00, 1.34),
    (11.00, -10.00, 1.34),
    (11.00, -8.00, 1.34),
    (11.00, -6.00, 1.34),
    (11.00, -4.00, 1.34),
    (11.00, -2.00, 1.34),
    (11.00, 2.00, 1.34),
    (11.00, 4.00, 1.34),
    (11.00, 6.00, 1.34),
    (11.00, 8.00, 1.34),
    (11.00, 10.00, 1.34),
    (11.00, 12.00, 1.34),
    (13.00, -12.00, 1.55),
    (13.00, -10.00, 1.55),
    (13.00, -8.00, 1.55),
    (13.00, -6.00, 1.55),
    (13.00, -4.00, 1.55),
    (13.00, -2.00, 1.55),
    (13.00, 2.00, 1.55),
    (13.00, 4.00, 1.55),
    (13.00, 6.00, 1.55),
    (13.00, 8.00, 1.55),
    (13.00, 10.00, 1.55),
    (13.00, 12.00, 1.55),
    (15.00, -12.00, 1.75),
    (15.00, -10.00, 1.75),
    (15.00, -8.00, 1.75),
    (15.00, -6.00, 1.75),
    (15.00, -4.00, 1.75),
    (15.00, -2.00, 1.75),
    (15.00, 2.00, 1.75),
    (15.00, 4.00, 1.75),
    (15.00, 6.00, 1.75),
    (15.00, 8.00, 1.75),
    (15.00, 10.00, 1.75),
    (15.00, 12.00, 1.75),
    (17.00, -14.00, 7.08),
    (17.00, -12.00, 1.95),
    (17.00, -12.00, 7.08),
    (17.00, -10.00, 1.95),
    (17.00, -10.00, 6.63),
    (17.00, -8.00, 1.95),
    (17.00, -8.00, 6.63),
    (17.00, -6.00, 1.95),
    (17.00, -6.00, 6.63),
    (17.00, -4.00, 1.95),
    (17.00, -2.00, 1.95),
    (17.00, -2.00, 6.63),
    (17.00, 0.00, 6.63),
    (17.00, 2.00, 1.95),
    (17.00, 2.00, 6.63),
    (17.00, 4.00, 1.95),
    (17.00, 6.00, 1.95),
    (17.00, 6.00, 6.63),
    (17.00, 8.00, 1.95),
    (17.00, 8.00, 6.63),
    (17.00, 10.00, 1.95),
    (17.00, 10.00, 6.63),
    (17.00, 12.00, 1.95),
    (17.00, 12.00, 7.08),
    (17.00, 14.00, 7.08),
    (19.00, -12.00, 2.15),
    (19.00, -10.00, 2.15),
    (19.00, -8.00, 2.15),
    (19.00, -8.00, 7.52),
    (19.00, -6.00, 2.15),
    (19.00, -6.00, 7.52),
    (19.00, -4.00, 2.15),
    (19.00, -2.00, 2.15),
    (19.00, 2.00, 2.15),
    (19.00, 4.00, 2.15),
    (19.00, 6.00, 2.15),
    (19.00, 6.00, 7.52),
    (19.00, 8.00, 2.15),
    (19.00, 8.00, 7.52),
    (19.00, 10.00, 2.15),
    (19.00, 12.00, 2.15),
    (21.00, -10.00, 2.23),
    (21.00, -6.00, 2.23),
    (21.00, -4.00, 2.23),
    (21.00, 4.00, 2.23),
    (21.00, 6.00, 2.23),
    (21.00, 10.00, 2.23)
]

SETUPS = {
    "RunA1": (["LS1"], list(MEASUREMENT_POINTS.values()) + SEAT_GRID),
    "RunA2": (["LS1"], list(MEASUREMENT_POINTS.values())),
    "RunB": (["LS7"], [MEASUREMENT_POINTS["MP1"]]),
}


def pach_points(kind):
    name = "Acoustical Source" if kind == "source" else "Acoustical Receiver"
    return [o for o in rs.AllObjects() if rs.IsPoint(o) and rs.ObjectName(o) == name]


def coords(o):
    p = rs.PointCoordinates(o)
    return (p.X, p.Y, p.Z)


def near(a, b):
    return max(abs(a[i] - b[i]) for i in range(3)) <= TOL


def label(xyz):
    for table in (SOURCES, MEASUREMENT_POINTS):
        for k, v in table.items():
            if near(xyz, v):
                return k
    for v in SEAT_GRID:
        if near(xyz, v):
            return "seat"
    return "not a BRAS position"


def report():
    s, r = pach_points("source"), pach_points("receiver")
    print("Sources (%d):" % len(s))
    for o in s:
        x, y, z = coords(o)
        print("  (%.2f, %.2f, %.2f)  %s" % (x, y, z, label((x, y, z))))
    named = [o for o in r if label(coords(o)) != "seat"]
    print("Receivers (%d): %d seat-grid points, plus:" % (len(r), len(r) - len(named)))
    for o in named:
        x, y, z = coords(o)
        print("  (%.2f, %.2f, %.2f)  %s" % (x, y, z, label((x, y, z))))


def apply(setup):
    want_s = [SOURCES[k] for k in SETUPS[setup][0]]
    want_r = SETUPS[setup][1]
    for kind, want, command in (("source", want_s, "Insert_Source"), ("receiver", want_r, "Insert_Receiver")):
        have = pach_points(kind)
        keep = [o for o in have if any(near(coords(o), w) for w in want)]
        rs.DeleteObjects([o for o in have if o not in keep])
        for w in want:
            if not any(near(coords(o), w) for o in keep):
                rs.Command("_%s %.3f,%.3f,%.3f" % (command, w[0], w[1], w[2]), False)


mode = rs.GetString("Mode", "Print", ["Print", "RunA1", "RunA2", "RunB"])
if mode in SETUPS:
    rs.EnableRedraw(False)
    apply(mode)
    rs.EnableRedraw(True)
if mode:
    report()
