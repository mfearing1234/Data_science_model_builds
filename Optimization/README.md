# Operations Research — Linear & Mixed-Integer Optimization

Business decision problems formulated as **linear programs (LP)** and **mixed-integer linear programs (MILP)**, then solved with [PuLP](https://coin-or.github.io/pulp/) and the open-source CBC solver. Each notebook follows the same structure: define the problem → decision variables → objective → constraints → solve → interpret.

## Notebooks

### 1. Multi-Echelon Supply Chain Network — `Multi_echelon_supplychain.ipynb`

**Problem:** A company ships product from 2 plants → 3 distribution centers → 4 customers. Plants cannot ship directly to customers. Minimize total shipping cost.

- **Decision variables:** flow on every plant→DC and DC→customer lane (continuous)
- **Constraints:** plant supply capacity, DC throughput capacity, flow conservation at each DC (inflow = outflow), customer demand satisfaction
- **Implementation:** wrapped in a reusable `supplychain_optimization_model` class that takes the network data as inputs, so the same model runs on any network size
- **Result:** optimal solution with **total cost of $2,230**, meeting all 290 units of demand

### 2. Production Planning — `Production_planning.ipynb`

**Problem:** A factory makes Products A ($40 profit) and B ($30 profit) subject to limited labor hours, raw material, and a demand cap on A. Maximize profit.

- **Decision variables:** integer units of each product
- **Constraints:** labor (100 hrs), raw material (180 lbs), max 40 units of A
- **Result:** optimal plan is **90 units of B, 0 of A**, for **$2,700 profit**. Although A has the higher unit profit, B earns more per pound of raw material ($15 vs. $13.33). Raw material is the binding constraint (all 180 lbs used, with 10 labor hours to spare), so the optimum shifts entirely to B.

### 3. Cinema Showtime & Revenue Optimization — `Movie_Revenue_and_Showtimes.ipynb`

**Problem:** Decide how many showtimes to schedule for each of five films, and how many tickets to sell, to maximize net revenue.

- **Decision variables:** integer showtimes and tickets sold per movie
- **Constraints:** demand limits, 250 seats per showtime, at least one showing per film, total screen availability, and total running time per day
- **Result:** optimal schedule yielding **$20,993 net revenue**, giving every film one showing and the extra capacity to the highest-priced, highest-demand film (Sci-Fi Odyssey, 3 showtimes)

## How to run

```bash
pip install -r ../requirements.txt
jupyter notebook
```

## Skills demonstrated

Mathematical modeling · LP / MILP formulation · network flow · capacity planning · PuLP · interpreting binding constraints
