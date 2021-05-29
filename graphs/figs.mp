%% metapost for diagrams showing the setups
prologues := 3;
%defaultscale := 12pt / fontsize defaultfont;
input boxes;
input TEX;

numeric pic_h, pic_w;

defaultfont := "phvr8g";

verbatimtex
 \font\tenrm=phvr8g at 9pt
 \font\sevenrm=phvr8g at 7pt
 \font\fiverm=phvr8g at 5pt
 \textfont0=\tenrm
 \scriptfont0=\sevenrm
 \scriptscriptfont0=\fiverm
 \def\rm{\fam=0 \tenrm}
 \rm
etex

filenametemplate "%j-%c.eps";

def send(suffix a, b) =
    drawarrow subpath(.1,.9) of (a.c -- b.c cutafter bpath b cutbefore bpath a);
enddef;

def send_dash(suffix a, b) =
    drawarrow a.c -- b.c cutafter bpath b cutbefore bpath a dashed evenly scaled 1.5;
enddef;

def conn(suffix a, b) =
    drawdblarrow subpath(.1,.9) of (a.c -- b.c cutafter bpath b cutbefore bpath a);
enddef;

def link(suffix a, b) =
    draw subpath(.1,.9) of (a.c -- b.c cutafter bpath b cutbefore bpath a);
enddef;

beginfig(1)
    begingroup
    pickup pencircle scaled .6pt;
    numeric u, sep, ang;
    numeric w, h;
    pair e[];
    u = 2.5cm;
    ang = 30;

    circleit.n0(btex 0 etex);
    circleit.n1(btex 1 etex);
    circleit.n2(btex 2 etex);
    circleit.n4(btex 4 etex);
    circleit.n5(btex 5 etex);
    circleit.n6(btex 6 etex);

    circleit.s0(btex  etex);
    circleit.s1(btex  etex);
    circleit.s2(btex  etex);
    circleit.s3(btex  etex);
    circleit.s4(btex  etex);
    circleit.s5(btex  etex);
    circleit.s6(btex  etex);

    % positions
    n0.c = (0,0);
    n1.c = n0.c + 1.25 * u * dir (-150);
    n2.c = n0.c + 1.25 * u * dir (-30);

    s0.c = n1.c + 0.75 * u * dir (-150);
    n4.c = n1.c + 0.75 * u * dir (-30);

    n5.c = n2.c + 0.75 * u * dir (-150);
    n6.c = n2.c + 0.75 * u * dir (-30);

    s1.c = n4.c + 0.5 * u * dir (-150);
    s2.c = n4.c + 0.5 * u * dir (-30);

    s3.c = n5.c + 0.5 * u * dir (-150);
    s4.c = n5.c + 0.5 * u * dir (-30);

    s5.c = n6.c + 0.5 * u * dir (-150);
    s6.c = n6.c + 0.5 * u * dir (-30);

    drawboxed(n0, n1, n2, n4, n5, n6);
    link(n0, n1);
    link(n0, n2);
    link(n1, s0);
    link(n1, n4);
    link(n2, n5);
    link(n2, n6);
    link(n4, s1);
    link(n4, s2);
    link(n5, s3);
    link(n5, s4);
    link(n6, s5);
    link(n6, s6);

    endgroup;
endfig;

beginfig(2)
    begingroup
    pickup pencircle scaled .6pt;
    numeric u, sep, ang;
    numeric w, h;
    pair e[];
    u = 2.5cm;
    ang = 30;

    circleit.n0(btex 0 etex);
    circleit.n1(btex 1 etex);
    circleit.n2(btex 3 etex);
    circleit.n4(btex 2 etex);
    circleit.n5(btex 4 etex);
    circleit.n6(btex 5 etex);

    circleit.s0(btex  etex);
    circleit.s1(btex  etex);
    circleit.s2(btex  etex);
    circleit.s3(btex  etex);
    circleit.s4(btex  etex);
    circleit.s5(btex  etex);
    circleit.s6(btex  etex);

    % positions
    n0.c = (0,0);
    n1.c = n0.c + 1.25 * u * dir (-150);
    n2.c = n0.c + 1.25 * u * dir (-30);

    s0.c = n1.c + 0.75 * u * dir (-150);
    n4.c = n1.c + 0.75 * u * dir (-30);

    n5.c = n2.c + 0.75 * u * dir (-150);
    n6.c = n2.c + 0.75 * u * dir (-30);

    s1.c = n4.c + 0.5 * u * dir (-150);
    s2.c = n4.c + 0.5 * u * dir (-30);

    s3.c = n5.c + 0.5 * u * dir (-150);
    s4.c = n5.c + 0.5 * u * dir (-30);

    s5.c = n6.c + 0.5 * u * dir (-150);
    s6.c = n6.c + 0.5 * u * dir (-30);

    drawboxed(n0, n1, n2, n4, n5, n6);
    link(n0, n1);
    link(n0, n2);
    link(n1, s0);
    link(n1, n4);
    link(n2, n5);
    link(n2, n6);
    link(n4, s1);
    link(n4, s2);
    link(n5, s3);
    link(n5, s4);
    link(n6, s5);
    link(n6, s6);

    endgroup;
endfig;
end

