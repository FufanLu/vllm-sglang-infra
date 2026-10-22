package quota

import (
	"context"
	"time"

	"github.com/redis/go-redis/v9"
)

// Redis is a Limiter shared by every gateway replica. Each operation is one
// Lua script, so refill, check, and take are atomic across gateways.
type Redis struct {
	client *redis.Client
	prefix string
}

func NewRedis(url, prefix string) (*Redis, error) {
	opts, err := redis.ParseURL(url)
	if err != nil {
		return nil, err
	}
	return &Redis{client: redis.NewClient(opts), prefix: prefix}, nil
}

func (r *Redis) Ping(ctx context.Context) error { return r.client.Ping(ctx).Err() }
func (r *Redis) Close() error                   { return r.client.Close() }

// The bucket is a hash {tokens, ts_ms}. Both scripts refill first, using the
// Redis server clock so gateways with skewed clocks agree.
const refill = `
local cap = tonumber(ARGV[1])
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or cap
local ts = tonumber(b[2]) or now
tokens = math.min(cap, tokens + (now - ts) * cap / 60000)
`

var reserveScript = redis.NewScript(refill + `
local n = tonumber(ARGV[2])
local ok = 0
if tokens >= n or tokens >= cap then
  tokens = tokens - n
  ok = 1
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', now)
redis.call('PEXPIRE', KEYS[1], 120000)
if ok == 1 then return {1, 0} end
return {0, math.ceil((n - tokens) * 60000 / cap)}
`)

var adjustScript = redis.NewScript(refill + `
tokens = math.min(cap, tokens - tonumber(ARGV[2]))
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', now)
redis.call('PEXPIRE', KEYS[1], 120000)
return 1
`)

func (r *Redis) Reserve(ctx context.Context, tenant string, perMinute, n int64) (bool, time.Duration, error) {
	if perMinute <= 0 {
		return true, 0, nil
	}
	res, err := reserveScript.Run(ctx, r.client, []string{r.prefix + tenant}, perMinute, n).Int64Slice()
	if err != nil {
		return false, 0, err
	}
	return res[0] == 1, time.Duration(res[1]) * time.Millisecond, nil
}

func (r *Redis) Adjust(ctx context.Context, tenant string, perMinute, delta int64) error {
	if perMinute <= 0 || delta == 0 {
		return nil
	}
	return adjustScript.Run(ctx, r.client, []string{r.prefix + tenant}, perMinute, delta).Err()
}
