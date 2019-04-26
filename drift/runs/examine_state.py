import math

if __name__ == '__main__':

    vx, vy, w = 1, -1.5, 3
    v = math.sqrt(vx**2 + vy**2)
    r = v / w
    tilt = math.degrees(math.atan2(vy, vx))
    print({
        'V': v,
        'Radius': r,
        'Tilt angle': tilt,
    })
