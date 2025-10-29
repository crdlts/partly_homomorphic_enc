import os
import csv
import random
import time
import datetime as dt
from typing import Tuple

import torch
import torch.distributed as dist
from datetime import timedelta

import config
from phe import paillier
from phe.paillier import EncryptedNumber


def now():
    return dt.datetime.utcnow().strftime('%H:%M:%S')


def log(rank, *a):
    print(f"[{now()}][r{rank}]", *a, flush=True)

# сериализация Paillier
def pk_serialize(pk: paillier.PaillierPublicKey) -> int:
    return pk.n


def pk_deserialize(n: int) -> paillier.PaillierPublicKey:
    return paillier.PaillierPublicKey(n)


def enc_serialize(enc) -> Tuple[int, int]:
    return (enc.ciphertext(), enc.exponent)


def enc_deserialize(pk: paillier.PaillierPublicKey, t: Tuple[int, int]):
    c, e = t
    return EncryptedNumber(pk, c, e)

# обмен через broadcast
def send_obj(obj, src: int):
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)


def recv_obj(src: int):
    obj_list = [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]

Q = int(config.MPC_MODULO)
def rz():
    return random.randrange(0, Q)


def modq(x: int) -> int:
    return x % Q

# Тройки Бивера rank0
def beaver_triple_once_rank0(pk, sk):
    a = rz()
    enc_a = pk.encrypt(a) # E(a) = g^a * r^n mod n^2
    send_obj(enc_serialize(enc_a), src=0) # отправляем Enc(a) стороне 1

    a2 = recv_obj(src=1) # получаем долю a2 (маска a)
    a1 = modq(a - a2)

    b1 = recv_obj(src=1) # получаем долю b1
    enc_t = enc_deserialize(pk, recv_obj(src=1)) # Enc(t), где t = ab + r
    t = modq(sk.decrypt(enc_t))

    c1 = rz() # делим c=ab на c1+c2
    s  = modq(t - c1) # s = (ab+r) - c1
    send_obj(s, src=0) # отправляем s, чтобы сторона 1 сняла r

    return (a1, b1, c1)

# Тройки Бивера rank1
def beaver_triple_once_rank1(pk):
    enc_a = enc_deserialize(pk, recv_obj(src=0)) # Enc(a)

    a2 = rz(); b = rz(); r = rz(); b2 = rz()
    b1 = modq(b - b2)

    enc_t = (enc_a * int(b)) + pk.encrypt(int(r)) # ab+r
    send_obj(a2, src=1) # отдаём маску a2
    send_obj(b1, src=1) # отдаём маску b1
    send_obj(enc_serialize(enc_t), src=1) # отдаём Enc(ab+r)

    s  = recv_obj(src=0) # s = (ab+r) - c1
    c2 = modq(s - r) # снимаем r и получаем свою долю c2

    return (a2, b2, c2)


def write_csv(path, triples):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['a','b','c'])
        for a, b, c in triples: w.writerow([int(a), int(b), int(c)])


def main():
    # параметры
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '2'))
    master_addr = os.environ.get('MASTER_ADDR', 'p1')
    master_port = os.environ.get('MASTER_PORT', '29500')
    backend = os.environ.get('DIST_BACKEND', 'gloo')
    num_triples = int(os.environ.get('NUM_TRIPLES', '16'))

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    log(rank, f"start (world={world}, backend={backend})")
    dist.init_process_group(backend=backend, rank=rank, world_size=world,
                            timeout=timedelta(seconds=180))
    log(rank, "dist init ok")

    # рассылаем public key, secret key только у rank0
    if rank == 0:
        t0 = time.time()
        pk, sk = paillier.generate_paillier_keypair(n_length=int(config.PAILLIER_KEY_SIZE))
        log(rank, f"paillier keygen {int((time.time()-t0)*1000)}ms")
        send_obj([pk_serialize(pk)], src=0)
    else:
        pk_n = recv_obj(src=0)[0]
        pk, sk = pk_deserialize(pk_n), None
        log(rank, "got pk")

    log(rank, f"triples: generating {num_triples}")
    triples = []
    for i in range(num_triples):
        triples.append(beaver_triple_once_rank0(pk, sk) if rank == 0 else beaver_triple_once_rank1(pk))
    log(rank, "triples: done")

    out_dir = os.environ.get('OUT_DIR', '/data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'p1.csv' if rank == 0 else 'p2.csv')
    write_csv(out_path, triples)
    log(rank, f"written {len(triples)} -> {out_path}")

    dist.barrier()
    dist.destroy_process_group()
    log(rank, "exit")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
