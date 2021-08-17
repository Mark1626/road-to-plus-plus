#!/usr/bin/env perl
use strict;

my %sym_occurance = ();

while (<>) {
  chomp;
  next if $_ =~ /^#/;

  if (/^\s*(\S.+?)\s+(\d+)\/*(\d+)*\s+/) {
    my ($comm, $pid, $tid) = ($1, $2, $3);
    if (not $tid) {
			$tid = $pid;
			$pid = "?";
		}

    if (/(\S+):\s*(\S+)\s+(\S+)/) {
      my ($event, $ip, $sym) = ($1, $2, $3);

      $_ = <>;
      chomp;
      my $srcline;
      if (/\s+(\S+:\d+)/) {
        $srcline = $1;
      } else {
        $srcline = 'Unknown';
        $sym = '?';
        next;
      }

      if (exists $sym_occurance{$srcline}) {
        $sym_occurance{$srcline}{occ}++;
      } else {
        $sym_occurance{$srcline}{symbol} = $sym;
        $sym_occurance{$srcline}{occ} = 1;
      }
    }
  } else {
    print STDERR "Unknown line";
  }
}

foreach my $k (sort { $a cmp $b } keys %sym_occurance) {
	print "$k $sym_occurance{$k}{symbol} $sym_occurance{$k}{occ}\n";
}
