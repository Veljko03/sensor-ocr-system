"""Experimental Festo geometry/structure candidate. Never used by production."""
import os
import cv2
import numpy as np
import real_meter_ocr as baseline

MODE = os.environ.get('OCR_CANDIDATE_MODE', 'dynamic')
PATTERNS = baseline.REAL_SEVEN_SEGMENT_PATTERNS


def candidates(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b,g,r = cv2.split(image.astype(np.int16))
    mask = cv2.inRange(hsv, np.array([85,35,40]), np.array([150,255,255]))
    mask |= np.uint8((b-(g+r)//2 > 18))*255
    scale = max(3, round(min(image.shape[:2])*.012)) | 1
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((scale,scale),np.uint8),iterations=2)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ranked=[]
    for contour in contours:
        area=cv2.contourArea(contour);rect=cv2.minAreaRect(contour)
        rw,rh=rect[1]
        if not .0008<=area/(image.shape[0]*image.shape[1])<=.50 or min(rw,rh)<12:continue
        aspect=min(rw,rh)/max(rw,rh)
        if aspect < .55:continue
        hull=cv2.convexHull(contour)
        for epsilon in (.01,.02,.03,.04,.05):
            polygon=cv2.approxPolyDP(hull,epsilon*cv2.arcLength(hull,True),True)
            if len(polygon)==4:break
        else:continue
        quad_area=cv2.contourArea(polygon)
        if quad_area<=0:continue
        fill=area/quad_area
        if fill<.65:continue
        quad=polygon.reshape(4,2).astype(np.float32)
        if len(np.unique(baseline.order_points(quad),axis=0))!=4:continue
        screen=baseline.perspective_warp(image,quad)
        screen_hsv=cv2.cvtColor(screen,cv2.COLOR_BGR2HSV)
        blue=(screen_hsv[:,:,0]>=85)&(screen_hsv[:,:,0]<=150)&(screen_hsv[:,:,1]>=50)&(screen_hsv[:,:,2]>=30)
        blue_fraction=float(blue.mean())
        ranked.append((area*(.5+fill),screen,{'aspect':round(aspect,4),'contour_fill':round(fill,4),'blue_fraction':round(blue_fraction,4),'quad_epsilon':epsilon}))
    return sorted(ranked,key=lambda row:row[0],reverse=True)[:3]


def crop_glyphs(screen):
    """Locate a dominant wide digit row, then require exactly three separated glyphs."""
    h,w=screen.shape[:2]
    b,g,r=cv2.split(screen.astype(np.float32))
    score=.5*r+.5*g-.35*b
    area=score[round(.18*h):round(.72*h),round(.28*w):round(.99*w)]
    normalized=cv2.normalize(area,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    normalized=cv2.GaussianBlur(normalized,(3,3),0)
    _,mask=cv2.threshold(normalized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    counts=np.count_nonzero(mask,axis=1).astype(float)
    if not counts.max():raise ValueError('No digit row')
    active=np.uint8(counts>max(6,counts.max()*.25))[:,None]
    active=cv2.morphologyEx(active,cv2.MORPH_CLOSE,np.ones((9,1),np.uint8)).ravel()
    starts=np.flatnonzero(np.diff(np.r_[0,active,0])==1)
    ends=np.flatnonzero(np.diff(np.r_[0,active,0])==-1)
    spans=[(start,end) for start,end in zip(starts,ends) if .17*h<=end-start<=.46*h]
    if not spans:raise ValueError('Invalid digit row height')
    y1,y2=max(spans,key=lambda pair:counts[pair[0]:pair[1]].sum())
    # Include thin extremal segments; suppress decimal dots by their small area.
    y1=max(0,y1-3);y2=min(mask.shape[0],y2+3)
    band=mask[y1:y2]
    n,labels,stats,_=cv2.connectedComponentsWithStats(band)
    clean=np.zeros_like(band)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=max(12,band.size*.004):clean[labels==i]=255
    col=np.count_nonzero(clean,axis=0)
    active=np.uint8(col>max(2,clean.shape[0]*.07))[None,:]
    active=cv2.morphologyEx(active,cv2.MORPH_CLOSE,np.ones((1,5),np.uint8)).ravel()
    starts=np.flatnonzero(np.diff(np.r_[0,active,0])==1)
    ends=np.flatnonzero(np.diff(np.r_[0,active,0])==-1)
    spans=[(s,e) for s,e in zip(starts,ends) if e-s>=4]
    if len(spans)!=3:raise ValueError('Expected three separate glyphs, found '+str(len(spans)))
    glyphs=[]
    for x1,x2 in spans:
        glyph=clean[:,max(0,x1-1):min(clean.shape[1],x2+1)]
        points=cv2.findNonZero(glyph)
        if points is None:raise ValueError('Empty glyph')
        x,y,bw,bh=cv2.boundingRect(points)
        if bh < band.shape[0]*.65:raise ValueError('Short glyph')
        glyphs.append(glyph[y:y+bh,x:x+bw])
    heights=[g.shape[0] for g in glyphs]
    if min(heights)/max(heights)<.75:raise ValueError('Inconsistent glyph heights')
    return glyphs,{'row_y':(round(.18*h)+y1,round(.18*h)+y2),'columns':spans}


def decode_glyph(glyph):
    h,w=glyph.shape[:2]
    ink=float(np.mean(glyph>0))
    if w/h<.32:
        # A one needs two vertical limbs spanning nearly the full glyph height.
        rows=np.mean(glyph>0,axis=1)
        if ink>=.45 and np.mean(rows>.25)>=.85:return '1',.95,{'narrow':True,'ink':round(ink,4)}
        return None,0.,{'reason':'Invalid narrow glyph'}
    if not .32<=w/h<=.95 or not .12<=ink<=.65:return None,0.,{'reason':'Glyph shape','ink':ink,'aspect':w/h}
    mask=cv2.resize(glyph,(80,120),interpolation=cv2.INTER_AREA)>127
    rois=[(.18,0,.82,.17),(.70,.12,1,.44),(.70,.56,1,.88),(.18,.83,.82,1),(0,.56,.30,.88),(0,.12,.30,.44),(.18,.42,.82,.58)]
    occupancies=[float(mask[round(y1*120):round(y2*120),round(x1*80):round(x2*80)].mean()) for x1,y1,x2,y2 in rois]
    holes=[float(mask[round(y1*120):round(y2*120),round(.36*80):round(.64*80)].mean()) for y1,y2 in ((.22,.36),(.64,.78))]
    observed=tuple(int(v>=.30) for v in occupancies)
    matching=[d for d,p in PATTERNS.items() if tuple(p)==observed]
    debug={'pattern':observed,'occupancies':[round(v,3) for v in occupancies],'holes':[round(v,3) for v in holes],'ink':round(ink,3),'aspect':round(w/h,3)}
    if len(matching)!=1 or max(holes)>.22:return None,0.,dict(debug,reason='Segment structure or filled interior')
    digit=matching[0]
    cost=sum((v-float(p))**2 for v,p in zip(occupancies,PATTERNS[digit]))/7
    return digit,1-cost,debug


def read_real_meter_image(image,confidence_threshold=.45):
    if image is None or image.size==0:return None,{'ok':False,'reason':'Empty image'}
    rejected=[];accepted=[]
    for _,screen,geometry in candidates(image):
        if MODE=='quad':
            value,reading=baseline.read_three_digit_value(screen,confidence_threshold)
        else:
            if geometry['blue_fraction']<.60:
                rejected.append(dict(geometry,reason='Background is not predominantly blue'));continue
            try:
                glyphs,row=crop_glyphs(screen)
                decoded=[decode_glyph(glyph) for glyph in glyphs]
                reading={'localization':row,'digits':[{'digit':d,'confidence':c,'debug':dbg} for d,c,dbg in decoded]}
                if any(d is None or c<confidence_threshold for d,c,_ in decoded):value=None
                else:value=int(''.join(d for d,_,_ in decoded))/100
            except ValueError as error:value=None;reading={'reason':str(error)}
        if value is None:rejected.append(dict(geometry,reading=reading))
        else:accepted.append((value,geometry,reading))
    if len(accepted)!=1:return None,{'ok':False,'reason':'No unique structurally valid display','accepted_candidates':len(accepted),'rejected':rejected}
    value,geometry,reading=accepted[0]
    return value,{'ok':True,'mode':MODE,'geometry':geometry,'reading':reading}
