scalePosWeight = 0
while  scalePosWeight <= 100:
    eta = 0.1
    while eta <= 1:
        depth = 5
        while depth <= 5:
            subSample = 0.1
            while subSample <= 1:
                min_child_weight = 1
                while min_child_weight <= 10:
                    print "scalePosWeight => "+str(scalePosWeight)+" eta => "+str(eta)+" depth =>"+str(depth)+" subSample =>"+str(subSample)+" min_child_weight =>"+str(min_child_weight)
                    min_child_weight = min_child_weight + 1
                subSample = subSample + 0.1
            depth = depth + 1
        eta = eta + 0.1
    scalePosWeight = scalePosWeight + 1