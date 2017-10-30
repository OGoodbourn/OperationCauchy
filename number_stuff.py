"""
Various functions of mathematical purpose.  Some of them could, just for example's sake, be used to solve the first 30
problems on a well-known mathematical programming challenge website.  Others can be used to learn about RSA encryption,
but should by no means be used to encrypt anything important!
"""


def factors(n):
    """Yield the factors of the integer n."""
    k = 1
    while k * k < n:
        if n % k == 0:
            yield k
            yield n // k
        k += 1
    if k * k == n:
        yield k


def sieve(n):
    """Return a list of length n+1 with entries, True (at prime indices) and False (otherwise)."""
    # Create a list of True/False values (all true, initially), then step through marking as false all those whose
    # index is not prime, using the sieve of Eratosthenes method.
    my_sieve = [False, False] + [True] * (n - 1)  # 0 and 1 are not prime
    for k in range(n + 1):
        # If k is prime, mark each multiple of k as composite.
        if my_sieve[k]:
            for l in range(2 * k, n + 1, k):
                my_sieve[l] = False
    return my_sieve


def is_prime(n):
    """Return True if num is prime and False if num is composite."""
    return sieve(n)[n]


def primes_up_to_nth(n):
    """Return a list of the first n prime numbers (not to be confused with a list of primes up to n)."""
    from math import sqrt
    DEFINITELY_ARBITRARILY_CHOSEN_BIG_NUMBER = 104744
    # Use sieve to create a list, primes_list, of primes less than some relatively large number.
    my_sieve = sieve(DEFINITELY_ARBITRARILY_CHOSEN_BIG_NUMBER)
    primes_list = []
    for k in range(DEFINITELY_ARBITRARILY_CHOSEN_BIG_NUMBER):
        if my_sieve[k]:
            primes_list.append(k)
    # If we already found the nth prime, return the list up to that point.
    if n <= (len(primes_list)):
        return primes_list[:n]
    # If we want primes higher than len(my_sieve), then start adding to our list one at a time until we have enough.
    # First set m to be 1 higher than the last (hence largest) entry in our list of primes, then test m for primality
    # and keep doing this until we have n primes.
    m = primes_list[-1] + 1
    while len(primes_list) < n:
        for p in primes_list:  # For each prime we found so far...
            if p < int(sqrt(m)) + 1:  # ...up to the square root of m...
                if m % p == 0:  # ...check if m is divisible by that prime.
                    break  # If so, m is composite, so end the for loop so we can move to a new m.
            # If p gets as high as the square root of m, we may as well add m to the list, as it has to be prime.
            else:
                primes_list.append(m)
                break  # Now, since it would waste time to finish iterating for this m, end the loop.
        m += 1
    return primes_list


def fib(n):
    """Return the nth term of the Fibonacci sequence starting 1, 1, 2, ..."""
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


def fibonacci_generator():
    """A generator for the Fibonacci sequence 0, 1, 1, 2, ... .  When iterated (e.g. "for n in fibonacci():") it steps through the sequence
    one Fibonacci number at a time.  Thus, one needs to provide a stop condition, or else it will loop endlessly!

    Examples:

    The following snippet would print Fibonacci numbers, one per line -- for ever!

    for n in fibonacci_generator():
        print(n, "\n")

    ...whilst the following would terminate once the numbers get above 100.

    for m in fibonacci_generator():
        print(m, "\n")
        if m > 100:
            break
    """
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b  # Simultaneous assignment handles the temporary value issue


def find_order_of_magnitude(f):
    """Return the order of magnitude of f."""
    if f == 0:
        return 0
    count = 0
    # o.o.m. >= 0
    if abs(f) >= 1:
        while abs(f) >= 1:
            f /= 10
            count += 1
        return count - 1
    # o.o.m. <0
    else:
        while abs(f) < 1:
            f *= 10
            count += 1
        return -count


def prime_factors(num):
    """
    Given an non-negative integer, num, not equal to 0 or 1 return a list of two lists: firstly a list of prime
    factors, secondly a list of their multiplicities.
    """
    from math import sqrt
    factors = []

    if num == 0 or num == 1:
        raise ValueError("Cannot factorise 0 or 1.")

    # In the following, we create a list whose entries alternate: prime, multiplicity, prime, multiplicity, ...
    # At the end, this is split into a list of primes and a list of corresponding multiplicities.

    # First strip out any/all multiples of 2.
    two_count = 0
    while num % 2 == 0:
        two_count += 1
        num /= 2
    factors.append(2)
    factors.append(two_count)

    # Now num is odd, so needn't consider even numbers.  Only need to go up to the sqrt of the number: if num has
    # no prime factor less than its sqrt, num must itself be prime.  Note also: since we step through in order, we'll
    # only pull out prime factors, as any composite number we reach is itself a product of smaller primes already
    # dealt with, so cannot divide num.
    for i in range(3, int(sqrt(num)) + 1, 2):
        i_count = 0
        while num % i == 0:
            i_count += 1
            num /= i
        factors.append(i)
        factors.append(i_count)
    # Now what's left (if anything) must be prime, so add that to the list, if present.
    if num > 2:  # In case num is itself 2, we already added it, so we don't do this step.
        factors.append(int(num))
        factors.append(1)

    prime_list = []
    mult_list = []
    for i in range(len(factors)):
        if i % 2 == 0:
            if factors[i + 1] != 0:
                prime_list.append(factors[i])
                mult_list.append(factors[i + 1])

    return [prime_list, mult_list]


def set_of_factors(num):
    """Given an integer num, return the set of all factors of num (including 1 and num)."""
    # Get the lists of prime factors and multiplicities.
    the_lists = prime_factors(num)
    (prime_list, mult_list) = (the_lists[0], the_lists[1])
    # Create a list of all the prime factors, repeated with their multiplicity.  E.g. for 12, get [2,2,3].
    primes_with_mult = []
    for k in range(len(prime_list)):
        primes_with_mult.extend([prime_list[k]] * mult_list[k])
    # Initialise the final list, beginning with the number 1.
    factor_list = [1]
    # For each combination of prime divisors, take product and add to list.
    for i in range(1, 2 ** len(primes_with_mult)):
        # Convert, e.g. i=5 to 0b111, then truncate to remove the 0b, then pad with leading zeros.  This creates
        # a string of 0s and 1s, of length the number of prime divisors.
        mask = str(bin(i)[2:]).zfill(len(primes_with_mult))
        # Now multiply together the prime divisors in primes_with_mult corresponding to a 1 in the mask.
        prod = 1
        for j in range(len(primes_with_mult)):
            if mask[j] == "1":
                prod = prod * primes_with_mult[j]
        factor_list.append(prod)
    return set(factor_list)


def sorted_list_of_factors(num):
    """Given an integer num, return a sorted list of all the factors of num (including 1 and num)."""
    return sorted(list(set_of_factors(num)))


def aliquot_sum(n):
    """Return the aliquot sum of n, i.e. the sum of all its proper divisors. E.g. for 9 we get 1 + 3 = 4."""
    proper_factors = set_of_factors(n)
    proper_factors.remove(n)
    return sum(proper_factors)


def is_amicable(n):
    """Return True if n is amicable and False otherwise."""
    if aliquot_sum(aliquot_sum(n)) == n and aliquot_sum(n) != n:
        return True
    else:
        return False


def is_abundant(n):
    """
    Return True if n is abundant and False otherwise.  We say n is abundant if the sum of its proper divisors
    is greater than n.
    """
    return aliquot_sum(n) > n


def ways_to_make_n_from_coins(n, list_of_coin_values):
    """
    Return the number of ways to make the value n from coins with the given values.

    The argument n should be an integer, and list_of_coin_values a list of unique integers.

    Example: n = 200, list_of_coin_values = [1, 2, 5, 10, 20, 50, 100, 200] produces the result 73682 (the number of
    ways of making £2 using any of the coins 1, 2, 5, 10, 20 or 50p and £1, £2.
    """
    list_of_coin_values = list_of_coin_values

    # Make list to hold the number of ways to make the index of the list, i.e. ways[i] will end up as n if there are
    # n ways to make i using the given coin values; initialize as [1,0,...,0] (1 way to make 0, others as yet unknown).
    ways = [1] + [0 for _ in range(n)]

    # Deal with / add in the different coin values/denominations one at a time (see below).
    for coin_value in list_of_coin_values:
        # Work up through the values, adding the (current) number of ways of making value i to the (current) number of
        # ways of making (i + coin_value) -- go as far up as this makes sense, i.e. don't overshoot the list length.
        # For example, if coin_value is 5 and there are n ways of making value i that we've counted so far, we're going
        # to have to account for n extra ways of making value (i + 5), now that we're allowed to use the coin value 5.

        # In more detail: beginning with no allowed coins, there are no ways of making any i greater than 0, and there's
        # exactly 1 way of making 0 (add 0 coins together).  If we now allow ourselves to use coins of value 10, we can
        # find out how many ways we can make value i by first seeing that we have 1 way to make 0, so however many ways
        # we found so far (0) to make the values 1--9 are unaffected, but from 10 upwards we must add 1 way to make that
        # value: thus ways[0 + 10] = ways[0 + 10] + ways[0], i.e. we get 1 more way to make 10; then
        # ways[1 + 10] = ways[1+10] + ways[1], i.e. we get 0 more ways to make 11, because we have currently no ways to
        # make 1.  Keep going up: ways[10 + 10] = ways[10 + 10] + ways[10], i.e. our zero ways of making 20 becomes
        # 0 + 1 way (which would be from adding 10 to 10).  We see that we only get non-zero ways of making multiples of
        #  10 on this first round.  If we had started with coin_value = 5, we'd get non-zero ways for multiples of 5; if
        #  we'd started with coin_value = 1 (as we will do) we'd get non-zero ways for all values.  It doesn't matter
        # which order we go through the coin values, however.  Back to our example where we started with 10:
        # We got ways = [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,...]
        # Now we try with coin_value = 2 .  We'll have to add in new ways of making values to take into account the
        # newer allowed coins.  We get ways[0+2] += ways[0] -- one new way of making 2 with the allowed coins. Then we
        # see ways[1+2] += ways[1] -- no new ways in this example.  ways[2+2] += ways[2], so 1 new way.  Continuing in
        # this way, we get up to i = 7 whereupon ways = [1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,...].  The next
        # step is then ways[8+2] += ways[8], so to our 1 in ways[10] we add the 1 in ways[8], which is what we should
        # have, as, using only coins of values 10 and 2, there are two ways to make 10: 10 and 2+2+2+2+2.
        for i in range(0, (n - coin_value) + 1):
            ways[i + coin_value] += ways[i]
    return ways[n]


def gcd(x, y):
    """Return the gcd of x and y, using the Euclidean algorithm"""
    while y != 0:
        (x, y) = (y, x % y)
    return x


def bezout(r, rr):
    """Return a tuple (r,s,t), where gcd(x,y) = r = sx + ty. Uses the extended Euclidean algorithm."""
    s, ss, t, tt = 1, 0, 0, 1
    while rr != 0:
        (r, rr, q) = (rr, r % rr, r // rr)
        (s, ss) = (ss, s - q * ss)
        (t, tt) = (tt, t - q * tt)
    return r, s, t


def mult_inv(n, m):
    """Return the multiplicative inverse of n modulo m if it exists (i.e. if (n,m) = 1); otherwise it does nothing."""
    d, n, _ = bezout(n, m)
    if d == 1:  # True when (n,m)=1; False o/w, and in that case don't do anything.
        return n % m


def miller_rabin(n, k, verbose=False):
    """
    A probabilistic primality test using the Miller-Rabin method.  Return False if n is composite and True if n is
    not shown to be composite (thus likely to be prime).  The parameter k determines the maximum number of
    bases to check.  If the optional parameter verbose is set to True, the process will be documented in the console.
    """
    import random
    if verbose:
        print("First test for divisibility by a number of small primes (in this case, those under 100.")
    small_primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    for p in small_primes:
        if n in small_primes:
            return True
        if n % p == 0 and n not in small_primes:
            if verbose:
                print("Divisible by a prime under 100, so composite.")
            return False
    if verbose:
        print("Since not divisible primes a prime under 100, begin with the Miller-Rabin test.")
    # Now determine positive integers d and s such that n - 1 =  d * (2 ** s), with s maximal (and d odd).
    d = n - 1
    s = 0
    while d % 2 == 0:
        d = d // 2
        s += 1
    if verbose:
        print("We write %d - 1 = %d * 2^%d" % (n, d, s))

    # The idea of the test is to pick a random base a in the set {1,...,n-1}, then examine the sequence
    # a**(n-1)=a**(d*2**s), a**(d*2**(s-1)), ..., a**(d*2), a**d (all modulo n), using Fermat's little theorem and
    # the fact that, working modulo a prime, there are no non-trivial square roots of unity.  If n is prime,
    # then a**(n-1) must be congruent to 1 modulo n; thus if it isn't congruent to 1, n must be composite.  This is
    # the Fermat test.  Each term of the sequence above is a square root of the previous term.  In case n is prime
    # so that the first term is congruent to 1 modulo n, the second term must be plus or minus 1 modulo n, by the
    # second observation above.  The Miller-Rabin test exploits this.

    # Create an empty set of previously picked bases.
    picked = set()
    # We can only pick at most n - 1 distinct numbers in the set {1, 2, ..., n-1}, so if k is too big,
    # first reduce it to this maximum.
    if k > (n - 1):
        k = n - 1

    while k > 0:
        # Pick a new a.
        a = random.randint(1, n - 1)
        if a in picked:
            # If a was already picked, continue from this loop to pick again (without decrementing k).
            continue

        # Now a is a new, unchecked base, so add it to the set picked.
        picked.add(a)
        if verbose:
            print("Picked base", a)

        # If a**d mod n is congruent to 1 or -1 mod n, then we learn nothing, because repeated squaring will always
        # give a value congruent to 1 modulo n, so we won't find any contradictions to n being prime. So pick
        # a new base and in this case decrement k.
        if pow(a, d, n) in {1, n - 1}:
            if verbose:
                print("pow(a,d,n) = pow(%d,%d,%d) is congruent to 1 or -1, \
and repeated squaring of this won't tell us if %d is composite" % (a, d, n, n))
            k -= 1
            continue

        # If we get to this point without "continuing", it must be the case that a**d is NOT congruent to 1 modulo n.
        if verbose:
            print("Since pow(%d,%d,%d) is not congruent to 1 or -1, we start checking \
the rest of the chain of squares" % (a, d, n))
        for i in range(1, s - 1):
            # Initialize the to_continue flag as False for this loop.
            to_continue = False

            if verbose:
                print("pow(a,d*2^%d,n) = pow(%d,%d*2**%d,%d) = %d" % (i, a, d, i, n, pow(a, d * 2 ** i, n)))
            # If a**(d*2**i) is congruent to 1 mod n, then n must be composite, because in the last round of the
            # loop (or just before the loop if this is the first round) we showed that a**(d*2**(i-1)) is NOT
            # congruent to plus or minus 1 modulo n.  For prime n, this can't happen: there are no non-trivial
            # square roots of 1 modulo prime n, but in the previous round we exhibited a non-trivial square root
            # of a**(d*2**i) (which is congruent to plus 1).
            if pow(a, d * 2 ** i, n) == 1:
                if verbose:
                    print("Thus n is composite, because the square root of this (%d) was not \
congruent to 1 or -1 modulo %d" % (pow(a, d * 2 ** (i - 1), n), n))
                return False
            elif pow(a, d * 2 ** i, n) == n - 1:
                if verbose:
                    print("Repeated squaring of this value will not tell us anything new as it is congruent to -1 \
modulo %d, so we move to new base." % n)
                to_continue = True
            # Using the same argument as earlier, we know we will learn nothing from this a, because repeated squaring
            # will now always give a value 1 modulo n, so we won't find any contradictions to n being prime. So pick a
            # new base.  Need to get around python's lack of labelled continues, though, hence the confusing way that's
            # dealt with here.
            if to_continue:
                break

        if pow(a, d * 2 ** (s - 1), n) not in {1, n - 1}:
            if verbose:
                print("%d^(%d-1)/2 = %d isn't congruent to plus or minus 1 modulo %d, so %d is composite \
by the Euler test." % (a, n, pow(a, d * 2 ** (s - 1), n), n, n))
            return False

        # If we made it through all s terms in the sequence, then we didn't learn anything, so move on to a new base.
        if verbose:
            print("We checked all the elements in the chain a^d, a^(2d), a^(4d), ..., a^((2^s)d) and couldn't \
conclude anything, so moving to new base.")
        k -= 1

    # If we made it through as many bases as we could (up to a maximum of k) without showing n to be composite,
    # return True, meaning n is probably prime.
    if verbose:
        print("We didn't find any Miller-Rabin witnesses, so probably prime.")
    return True


def get_prime_of_length(n):
    """Return a probable prime of bit length n."""
    import random
    found = False
    # Also putting in a probably-redundant check to make sure we don't loop forever -- because I haven't thought about
    # whether there could be a really big prime gap that could cause problems for some n.  Seems implausible for
    # practical purposes, though.
    loops = 1
    max = 2 ** (n - 1)  # NB: Using base 2, so 2^n - 2^(n-1) = 2^(n-1)
    bottom = 2 ** (n - 1)
    top = 2 ** n
    while not found and loops < max:
        p = random.randint(bottom, top)
        if p % 2 == 0:
            continue
        if miller_rabin(p, 1):
            found = True
        if found:
            continue
        loops += 1
    return p


def rsa_key_gen(length=1024):
    """
    Return list consisting of a private key (pq, d) and a public key (pq, e) such that pq has the given bit
    length (defaults to 1024).
    """
    from math import floor
    p = get_prime_of_length(floor((length + 1) / 2))
    found = False
    q = 1
    while not found and p != q:
        q = get_prime_of_length(floor((length + 1) / 2))
        if (p * q).bit_length() == length:
            found = True
    phi = (p - 1) * (q - 1)
    # Pick e appropriately.  It should be coprime to phi, but this is easily satisfied for prime e.
    if length >= 17:
        e = 65537
    elif length >= 8:
        e = 257
    elif length >= 4:
        e = 17
    elif length >= 2:
        e = 5
    else:
        e = 3
    d = mult_inv(e, phi)
    return [(p * q, d), (p * q, e)]  # [private key, public key]


def demonstrate_rsa(secret_integer):
    """Demonstrate RSA using artificially short bit length key for readability."""
    keys = rsa_key_gen(50)
    private = keys[0]
    print("Private key:", private)
    public = keys[1]
    print("Public key:", public)
    print("Raise %d to power %d mod %d" %(secret_integer, public[1], public[0]))
    print("Get encoded integer:")
    print(pow(secret_integer, public[1], public[0]))
    print("Raise the encoded integer %d to power %d mod %d" % (pow(secret_integer, public[1], public[0]),
                                                               private[1], private[0]))
    print("Get decoded integer:")
    print(pow(pow(secret_integer, public[1], public[0]), private[1], private[0]))
    print("A match?", pow(pow(secret_integer, public[1], public[0]), private[1], private[0]) == secret_integer)

