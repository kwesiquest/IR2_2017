'''Stopwords Corpus

This corpus contains lists of stop words for several languages.  These
are high-frequency grammatical words which are usually ignored in text
retrieval applications.

They were obtained from:
http://anoncvs.postgresql.org/cvsweb.cgi/pgsql/src/backend/snowball/stopwords/

The stop words for the Romanian language were obtained from:
http://arlc.ro/resources/

The English list has been augmented
https://github.com/nltk/nltk_data/issues/22

The German list has been corrected
https://github.com/nltk/nltk_data/pull/49

A Kazakh list has been added
https://github.com/nltk/nltk_data/pull/52'''

stopwords = '''i
me
my
myself
we
our
ours
ourselves
you
your
yours
yourself
yourselves
he
him
his
himself
she
her
hers
herself
it
its
itself
they
them
their
theirs
themselves
what
which
who
whom
this
that
these
those
am
is
are
was
were
be
been
being
have
has
had
having
do
does
did
doing
a
an
the
and
but
if
or
because
as
until
while
of
at
by
for
with
about
against
between
into
through
during
before
after
above
below
to
from
up
down
in
out
on
off
over
under
again
further
then
once
here
there
when
where
why
how
all
any
both
each
few
more
most
other
some
such
no
nor
not
only
own
same
so
than
too
very
s
t
can
will
just
don
should
now
d
ll
m
o
re
ve
y
ain
aren
couldn
didn
doesn
hadn
hasn
haven
isn
ma
mightn
mustn
needn
shan
shouldn
wasn
weren
won
wouldn'''


def _get_stopwords():
    stopwords_list = stopwords.split()
    stopwords_list.append('&')
    return stopwords_list

stopwords = _get_stopwords()


if __name__ == '__main__':
    pass
