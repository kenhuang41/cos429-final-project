import numpy as np
import cv2

def convolve_x(img, height = 9, sigma = 0.2, threshold = 30):
    Gx = np.array([1 / (sigma * np.sqrt(2 * np.pi)) * np.e**(-y**2 / (2 * sigma**2)) for y in range(-height//2+1, height//2+1)])
    Gx = np.expand_dims(Gx, axis = 0) / np.sum(Gx)
    Gx = Gx - np.ones(height) / height
    out = cv2.filter2D(img, -1, Gx)
    out[out > threshold] = 255
    return out

def convolve_y(img, height = 9, sigma = 1):
    Gy = np.array([1 / (sigma * np.sqrt(2 * np.pi)) * np.e**(-y**2 / (2 * sigma**2)) for y in range(-height//2+1, height//2+1)])
    Gy = np.expand_dims(Gy, axis = 1) / np.sum(Gy)
    Gy = Gy - np.ones(height) / height
    return cv2.filter2D(img, -1, Gy)

def detectLines(img, minVotes = 10, rRes = 1, rPrev = None, rRange = 30, tRes = np.pi / 180, tPrev = None, tRange = 0.10, minBrightness = 200, best = None, optimizer = None):
    height, width = img.shape
    ans = dict()
    
    if minVotes < 10:
        return np.array([[]])
    
    if optimizer == None:
        height_range = range(height)
        width_range = range(width)
    elif optimizer == 'bow':
        height_range = range(height//3, 3*height//4)
        width_range = range(width//4, 3*width//4)
    elif optimizer == 'fingerboard':
        height_range = range(height//2)
        width_range = range(width//4, 3*width//4)
        
    if tPrev == None:
        t_range = range(int(np.pi / tRes))
    else:
        t_range = range(int((tPrev - tRange/2) / tRes), int((tPrev + tRange/2) / tRes))
    
    for r in height_range:
        for c in width_range:
            if img[r,c] < minBrightness:
                continue
            for i in t_range:
                rho = (r * np.cos(i * tRes) + c * np.sin(i * tRes)) // rRes * rRes
                if rPrev != None and abs(rho - rPrev) > rRange / 2:
                    continue
                if rho in ans.keys():
                    ans[rho][i] = ans[rho].get(i, 0) + 1
                else:
                    ans[rho] = dict()
                    ans[rho][i] = 1
                    
    # create output array
    out = []
    for rho, d in ans.items():
        for theta, votes in d.items():
            if votes >= minVotes:
                out.append([rho, theta * tRes, ans[rho][theta]])
    if len(out) == 0:
        return detectLines(img, minVotes//3, rRes, rPrev, rRange, tRes, tPrev, tRange, minBrightness, best, optimizer)
    out = np.array(out)
    
    if best == None:
        return out[(-out)[:, 2].argsort()]
    return out[(-out)[:, 2].argsort()][:best]

def detect_bow(img, rPrev = None, tPrev = None, optimized = False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Fy = convolve_y(gray)
    lines = detectLines(Fy, minVotes = gray.shape[1] // 3, rRes = 1, rPrev = rPrev, tRes = np.pi / 180, tPrev = tPrev, tRange = 0.3, best = 1, optimizer = 'bow' if optimized else None)
    temp = img.copy()

    if not np.any(lines):
        return temp, None, None
    out_rho = None
    out_theta = None
    for l in lines:
        rho, theta, votes = l
        out_rho = rho
        out_theta = theta
        b = np.cos(theta)
        a = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(temp,(x1,y1), (x2,y2), (0,0,255), 2)

    return temp, out_rho, out_theta

def detect_fingerboard(img, rPrev = None, tPrev = None, optimized = False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Fx = convolve_x(gray)
    lines = detectLines(Fx, minVotes = gray.shape[0] // 3, rRes = 1, rPrev = rPrev, tRes = np.pi / 180, tPrev = tPrev, best = 1, optimizer = 'fingerboard' if optimized else None)
    temp = img.copy()

    if not np.any(lines):
        return temp, None, None
    out_rho = None
    out_theta = None
    for l in lines:
        rho, theta, votes = l
        out_rho = rho
        out_theta = theta
        b = np.cos(theta)
        a = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(temp,(x1,y1), (x2,y2), (255,0,0), 2)

    return temp, out_rho, out_theta

def detect_both(img, rPrev1 = None, rPrev2 = None, tPrev1 = None, tPrev2 = None, optimized = False):
    
    temp1, out_rho1, out_theta1 = detect_fingerboard(img, rPrev1, tPrev1, optimized)
    temp2, out_rho2, out_theta2 = detect_bow(temp1, rPrev2, tPrev2, optimized)
    return temp2, out_rho1, out_rho2, out_theta1, out_theta2
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = img.copy()

    # find fingerboard
    Fx = convolve_x(gray)
    lines = detectLines(Fx, minVotes = gray.shape[0] // 3, rRes = 1, rPrev = rPrev1, tRes = np.pi / 180, tPrev = tPrev1, best = 1, optimizer = 'fingerboard' if optimized else None)
    
    out_rho1 = None
    out_theta1 = None
    for l in lines:
        rho, theta, votes = l
        out_rho1 = rho
        out_theta1 = theta
        b = np.cos(theta)
        a = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(temp,(x1,y1), (x2,y2), (255,0,0), 2)
        
    # find bow
    Fy = convolve_y(gray)
    lines = detectLines(Fy, minVotes = gray.shape[1] // 3, rRes = 1, rPrev = rPrev2, tRes = np.pi / 180, tPrev = tPrev2, tRange = 0.3, best = 1, optimizer = 'bow' if optimized else None)
    
    out_rho2 = None
    out_theta2 = None
    for l in lines:
        rho, theta, votes = l
        out_rho2 = rho
        out_theta2 = theta
        b = np.cos(theta)
        a = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(temp,(x1,y1), (x2,y2), (0,0,255), 2)

    return temp, out_rho1, out_rho2, out_theta1, out_theta2