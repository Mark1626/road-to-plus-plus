#!/usr/bin/env perl
# split-process.pl <perf output>
# Usage perf script -F+pid | split-process.pl
use strict;

my %files = {};
my $pid = '';
my $comm = '';
my $tid = '';

while (<>) {
  chomp;
  next if $_ =~ /^#/;
  if (/^\s*(\S.+?)\s+(\d+)\/*(\d+)*\s+/) {
    ($pid, $tid) = ($2, $3);
    if (not $tid) {
			$tid = $pid;
			$pid = "?";
		}
  }
  open my $fh, '>>', "out-$pid.perf";
  print $fh $_ . "\n";
}
