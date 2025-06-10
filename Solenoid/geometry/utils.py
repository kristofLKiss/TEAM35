def append_unique_point(geo, x, y, points):
    """Appends a point to the geometry if it doesn't already exist."""
    if (x, y) not in points:
        points[(x, y)] = geo.AppendPoint(x, y)
    return points[(x, y)]

def append_unique_line(geo, p1, p2, lines, leftdomain, rightdomain):
    """Appends a line to the geometry if it doesn't already exist, in any direction."""
    if (p1, p2) not in lines and (p2, p1) not in lines:
        geo.Append(["line", p1, p2], leftdomain=leftdomain, rightdomain=rightdomain)
        lines.add((p1, p2))

def add_rectangle_with_edge_domains(geo, lines, points, p1, p2,
                   leftdomain, rightdomain_N, rightdomain_S, rightdomain_W, rightdomain_E):
    """Adds a rectangle with distinct boundary domains on each edge, avoiding duplicate points/lines."""
    pts = [
        append_unique_point(geo, p1[0], p1[1], points),
        append_unique_point(geo, p2[0], p1[1], points),
        append_unique_point(geo, p2[0], p2[1], points),
        append_unique_point(geo, p1[0], p2[1], points)
    ]

    append_unique_line(geo, pts[0], pts[1], lines, leftdomain, rightdomain_S)  # Bottom
    append_unique_line(geo, pts[1], pts[2], lines, leftdomain, rightdomain_E)  # Right
    append_unique_line(geo, pts[2], pts[3], lines, leftdomain, rightdomain_N)  # Top
    append_unique_line(geo, pts[3], pts[0], lines, leftdomain, rightdomain_W)  # Left

    print([pts, rightdomain_W, leftdomain, rightdomain_E])
