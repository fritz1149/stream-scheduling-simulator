from pyscipopt import Model
model = Model("Example")  # model name is optional
x = model.addVar("x")
y = model.addVar("y", vtype="INTEGER")
z = model.addVar("z")
model.setObjective(z)
model.addCons(2*x - y*y >= 0)
model.addCons(x*y == z)
model.optimize()
sol = model.getBestSol()

print("x: {}, y: {}, z:{}".format(sol[x], sol[y], sol[z]))