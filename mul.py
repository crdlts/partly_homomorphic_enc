import csv
import random 
import config
from collections import namedtuple

  
Triple = namedtuple('Triple', ['a', 'b', 'c'])


def load_triples(path) -> list[Triple]:
  with open(path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    return [Triple(*map(int, row)) for row in reader]
  

def test_triples(p1: list[Triple], p2: list[Triple]):
  if len(p1) != len(p2):
    raise ValueError(f'len(p1) = {len(p1)}, but len(p2)={len(p2)}')
  
  for i in range(len(p1)):
    x1 = random.randint(0, config.MPC_MODULO)
    x2 = (x1-random.randint(0, config.MPC_MODULO)) % config.MPC_MODULO

    y1 = random.randint(0, config.MPC_MODULO)
    y2 = (x1-random.randint(0, config.MPC_MODULO)) % config.MPC_MODULO

    r = ((x1+x2) * (y1+y2)) % config.MPC_MODULO

    d = ((x1-p1[i].a) + (x2-p2[i].a)) % config.MPC_MODULO
    e = ((y1-p1[i].b) + (y2-p2[i].b)) % config.MPC_MODULO

    z1 = p1[i].c + d*p1[i].b + e*p1[i].a + d*e
    z2 = p2[i].c + d*p2[i].b + e*p2[i].a
    z = (z1+z2) % config.MPC_MODULO

    if r != z:
      raise ValueError(f'({p1[i].a + p2[i].a}, {p1[i].b + p2[i].b}, {p1[i].c + p2[i].c} is a wrong triple!')


if __name__ == '__main__':
  p1_triples = load_triples('data/p1.csv')
  p2_triples = load_triples('data/p2.csv')

  test_triples(p1_triples, p2_triples)

  print('All triples are correct')


