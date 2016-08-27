l = [.1 .3 .6 .9];
ig_info = .9;
race_info = .9;
sig_dist = [.3 .2 .2 .3];
ft = [ig_info * race_info (1-ig_info) * race_info ig_info * (1-race_info) (1-ig_info) * (1-race_info);
    ig_info * (1-race_info) (1-ig_info) * (1-race_info) ig_info * race_info (1-ig_info) * race_info];
gt = [(1-ig_info) * race_info ig_info * race_info (1-ig_info) * (1-race_info) ig_info * (1-race_info);
    (1-ig_info) * (1-race_info) ig_info * (1-race_info) (1-ig_info) * race_info ig_info * race_info];

guilts = [sig_dist * gt(1,:)' sig_dist * gt(2,:)'];
inns = [sig_dist * ft(1,:)' sig_dist * ft(2,:)'];
pops = guilts ./ (guilts + inns);

cvx_begin
    variable x(4,4)
    maximize(1)
    subject to
        for s = 1:4
            sum(x(s,:)) == sig_dist(s);
        end
        for b = 1:4
            for t = 1:2
                x(:,b)' * gt(t,:)' == l(b) * (x(:,b)' * gt(t,:)' + x(:,b)' * ft(t,:)');
            end
        end
        ((ft(1,:) * x) * l')/inns(1) == ((ft(2,:) * x) * l')/inns(2);
        x >= 0;
cvx_end