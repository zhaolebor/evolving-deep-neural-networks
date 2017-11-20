from . import population
from . import species
from . import chromosome
from . import genome

module_pop = population.Population(chromosome.ModuleChromo)
blueprint_pop = population.Population(chromosome.BlueprintChromo, module_pop)

