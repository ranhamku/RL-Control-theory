import random, statistics

class Test:
    tries = 1 #number of attempts for this test
    successes = 1 # number of success for this test
    rate = 1 # the denominator of the success rate. 1/2, 1/4, 1/5 whatever
    def __init__(self, rate=1):
        self.rate=rate
    def run(self): # run the test
        self.tries += 1
        x = random.randint(1,self.rate)
        if x == 1:
            self.successes += 1
        return self

def choose(self,tests): # choose which test to run
    if random.randint(0,9) == 0: # one out of ten tests are random
        x = random.randint(0,len(tests)-1)
        tests[x].run()
        return None
    else: #go with whichever test has the best success ration
        bestRate = 0
       # bestTest = choose(tests)
        for test in tests:
            rate = test.successes/test.tries
            if rate > bestRate:
                bestRate = rate
                self.bestTest = test
        bestTest.run()
        return bestTest

def run1000Tests():
    winner = None #keep track of which test was selected as a "bestTest"
    windex = 0
    tests = [Test(2), Test(4), Test(5), Test(6)]
    random.shuffle(tests)
    for i in range(0,1000):
        bestTest = choose(self,ests)
        if bestTest and winner != bestTest: # if there has been a change in which test is te "bestTest"
            winner = bestTest # update the bestTest, but more importantly, update the index of which we decidded this was the best test
            windex = i
    return windex #the windex is the index of the last time that we deemed a new bestTest

windexes = []
for i in range(0,1000):
    windexes.append(run1000Tests()) #build an array of the windexes

print("Number of tests it took, on average, to figure out which test was performing the best and stick to it:")
print(statistics.mean(windexes))

print("\nMax number of tests to pick our winner:")
print(max(windexes))

