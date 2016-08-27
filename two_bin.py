from __future__ import division
ga = .42
gb = .52

def ps(g, r1, r2):
    p1 = (r2-g)/(r2/r1-1)
    return p1, p1*(1/r1-1)

def unfairness(g, r1, r2):
    p1, p2 = ps(g, r1, r2)
    return (p2*r1 + (1-g-p2)*r2)/(1-g)

def allocation(g, r1, r2):
    p1, p2 = ps(g, r1, r2)
    print '\t1\t2'
    print 'inn\t%.3f\t%.3f' % (p2, 1-g-p2)
    print 'gui\t%.3f\t%.3f' % (p1, g-p1)

if __name__ == '__main__':
    r1a, r2a = .1, .42
    unf = unfairness(ga, r1a, r2a)
    # r2b = .9
    # r1b = 1-((r2a-ga)*(1-r1a)/(1-ga) - r2a+r2b)*(1-gb)/(r2b-gb)
    r1b = .22
    r2b = (unf*(1-gb) - gb*(1-r1b))/(r1b-gb)
    print 'a bins', r1a, r2a
    print 'b bins', r1b, r2b
    print 'unfairness a', unfairness(ga, r1a, r2a)
    print 'unfairness b', unfairness(gb, r1b, r2b)
    print 'A'
    allocation(ga, r1a, r2a)
    print 'B'
    allocation(gb, r1b, r2b)
    p2a = ps(ga, r1a, r2a)[1]
    p2b = ps(gb, r1b, r2b)[1]
