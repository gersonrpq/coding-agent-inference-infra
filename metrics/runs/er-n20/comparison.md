| run | served/min | TTFT p50 | TTFT p99 | shed % | cached % | cached % same worker | cached % other worker | stickiness | calls on worker 0 % | decode tok/s | engine queue max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| affload | 54.0 | 1.7 | 7.6 | 41.9 | 87.7 | 92.3 | 41.5 | 0.88 | 50.6 | 29.6 | 2/1 |
| lb | 53.7 | 1.8 | 9.5 | 29.7 | 82.1 | 90.3 | 79.6 | 0.55 | 50.3 | 28.9 | 2/3 |
| shuffle | 41.7 | 6.2 | 19.3 | 51.4 | 79.5 | 89.5 | 78.9 | 0.46 | 41.6 | 27.4 | 1/7 |
