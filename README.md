# ice_betting
## intro to git.

The original idea was to use CLI to 
 1. Students clone repo with empty student ledger,  cat open the individial ledger, fill their name/group/bet amount/predicted date/id,etc. 
 2. Then they merge this back into a global ledger.
 3. The odds of each prediction depends on a set distriubution that gets ' bayesianly' updated wth the bets that have been placed (global ledger)
 4. The actual mapping from probablity to odd, is arbitrarly, the function that I choosed aproximatelty  maps (prob=0.1)-> (odds=1.2) and (prob=0.05)->(odds,4) ( kinda unfair)  5. Before placing the bet, student can pull the most recent ledger, use get_odds, and get the odds for a chosen date without placing the bet ( all of this on CLI)

I ran out of timeand could not to set up the CI/CD gH pipeline ( if it would even work), so **currently** everything works on the local repo ( see example.py).
The function that creates plots from the 'global ledger' work as well.

  
