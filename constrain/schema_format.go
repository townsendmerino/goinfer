package constrain

// strFormat is a JSON Schema string `format` the grammar enforces. Any other format value is an annotation, as JSON
// Schema 2020-12 defines it, and the string stays free.
type strFormat int

const (
	fmtNone     strFormat = iota
	fmtDate               // RFC 3339 full-date: YYYY-MM-DD, month 01-12, day 01-31
	fmtTime               // RFC 3339 full-time: HH:MM:SS[.frac] then Z or ±HH:MM (second 60 allowed for a leap second)
	fmtDateTime           // full-date, 'T' or 't', full-time
	fmtUUID               // RFC 4122 text form: 8-4-4-4-12 hex digits
	fmtIPv4               // dotted quad, each 0-255 with no leading zero
	fmtEmail              // dot-atom local part, '@', two or more dot-separated domain labels
)

// stringFormats maps a `format` value to the format the grammar enforces.
var stringFormats = map[string]strFormat{
	"date": fmtDate, "time": fmtTime, "date-time": fmtDateTime, "uuid": fmtUUID, "ipv4": fmtIPv4, "email": fmtEmail,
}

// A formatted string's frame keeps its acceptor state in fields a string frame does not otherwise use: count is the phase,
// pos the position within it, sel the previous byte, and num a running value (an ipv4 group's, an email's domain dots).
const (
	phDate = iota
	phSep
	phTime
	phTimeEnd
	phFrac
	phOffset
	phDone
)

func isDigit(b byte) bool { return b >= '0' && b <= '9' }

// atext is RFC 5322's atom character set, the bytes of a dot-atom email local part other than '.'.
func atext(b byte) bool {
	switch {
	case isDigit(b), b|0x20 >= 'a' && b|0x20 <= 'z':
		return true
	}
	switch b {
	case '!', '#', '$', '%', '&', '\'', '*', '+', '-', '/', '=', '?', '^', '_', '`', '{', '|', '}', '~':
		return true
	}
	return false
}

// fmtStart is a formatted string frame's state right after its opening quote.
func fmtStart(f *frame) {
	f.pos, f.num, f.sel = 0, 0, 0
	f.count = phDate
	if f.n.format == fmtTime {
		f.count = phTime
	}
}

// dateByte accepts b at position p of YYYY-MM-DD, given the previous byte.
func dateByte(p int, prev, b byte) bool {
	switch p {
	case 0, 1, 2, 3:
		return isDigit(b)
	case 4, 7:
		return b == '-'
	case 5:
		return b == '0' || b == '1'
	case 6:
		return (prev == '0' && b >= '1' && b <= '9') || (prev == '1' && b >= '0' && b <= '2')
	case 8:
		return b >= '0' && b <= '3'
	case 9:
		return (prev == '0' && b >= '1' && b <= '9') || (prev == '3' && (b == '0' || b == '1')) || (prev != '0' && prev != '3' && isDigit(b))
	}
	return false
}

// hhmmByte accepts b at position p of HH:MM (an hour 00-23, a minute 00-59), given the previous byte.
func hhmmByte(p int, prev, b byte) bool {
	switch p {
	case 0:
		return b >= '0' && b <= '2'
	case 1:
		return isDigit(b) && (prev != '2' || b <= '3')
	case 2:
		return b == ':'
	case 3:
		return b >= '0' && b <= '5'
	case 4:
		return isDigit(b)
	}
	return false
}

// fmtStep is a formatted string's content step: it accepts b when the string so far plus b is a prefix of a value of the
// format. The closing quote is fmtTerminal's to allow.
func fmtStep(f *frame, b byte) bool {
	prev := byte(f.sel)
	f.sel = int(b)
	switch f.n.format {
	case fmtUUID:
		if f.pos >= 36 {
			return false
		}
		ok := isHex(b)
		if f.pos == 8 || f.pos == 13 || f.pos == 18 || f.pos == 23 {
			ok = b == '-'
		}
		f.pos++
		return ok
	case fmtIPv4: // count: groups finished; pos: digits in this group; num: its value
		switch {
		case isDigit(b):
			if f.pos == 3 || (f.pos == 1 && f.num == 0) || f.num*10+int(b-'0') > 255 {
				return false
			}
			f.pos, f.num = f.pos+1, f.num*10+int(b-'0')
			return true
		case b == '.' && f.pos > 0 && f.count < 3:
			f.count, f.pos, f.num = f.count+1, 0, 0
			return true
		}
		return false
	case fmtEmail: // count 0: local part, 1: domain; pos: bytes in the current part or label; num: domain dots
		if f.count == 0 {
			switch {
			case atext(b):
				f.pos++
				return true
			case b == '.' && f.pos > 0 && prev != '.':
				f.pos++
				return true
			case b == '@' && f.pos > 0 && prev != '.':
				f.count, f.pos = 1, 0
				return true
			}
			return false
		}
		switch {
		case isDigit(b) || (b|0x20 >= 'a' && b|0x20 <= 'z'):
			f.pos++
			return true
		case b == '-' && f.pos > 0:
			f.pos++
			return true
		case b == '.' && f.pos > 0 && prev != '-':
			f.pos, f.num = 0, f.num+1
			return true
		}
		return false
	}
	return timeStep(f, prev, b)
}

// timeStep is date, time and date-time's step, phase by phase (frame.count).
func timeStep(f *frame, prev, b byte) bool {
	switch f.count {
	case phDate:
		if !dateByte(f.pos, prev, b) {
			return false
		}
		if f.pos++; f.pos == 10 {
			f.count, f.pos = phDone, 0
			if f.n.format == fmtDateTime {
				f.count = phSep
			}
		}
		return true
	case phSep:
		if b != 'T' && b != 't' {
			return false
		}
		f.count, f.pos = phTime, 0
		return true
	case phTime: // HH:MM:SS
		var ok bool
		switch f.pos {
		case 0, 1, 2, 3, 4:
			ok = hhmmByte(f.pos, prev, b)
		case 5:
			ok = b == ':'
		case 6:
			ok = b >= '0' && b <= '6'
		case 7:
			ok = isDigit(b) && (prev != '6' || b == '0')
		}
		if !ok {
			return false
		}
		if f.pos++; f.pos == 8 {
			f.count, f.pos = phTimeEnd, 0
		}
		return true
	case phTimeEnd, phFrac:
		switch {
		case f.count == phTimeEnd && b == '.':
			f.count, f.pos = phFrac, 0
			return true
		case f.count == phFrac && isDigit(b):
			f.pos++
			return true
		case f.count == phFrac && f.pos == 0:
			return false // a fraction needs a digit
		case b == 'Z' || b == 'z':
			f.count = phDone
			return true
		case b == '+' || b == '-':
			f.count, f.pos = phOffset, 0
			return true
		}
		return false
	case phOffset:
		if !hhmmByte(f.pos, prev, b) {
			return false
		}
		if f.pos++; f.pos == 5 {
			f.count = phDone
		}
		return true
	}
	return false
}

// fmtTerminal reports whether a formatted string may close here: the content is a whole value of its format.
func fmtTerminal(f *frame) bool {
	switch f.n.format {
	case fmtUUID:
		return f.pos == 36
	case fmtIPv4:
		return f.count == 3 && f.pos > 0
	case fmtEmail:
		return f.count == 1 && f.pos > 0 && f.num >= 1 && byte(f.sel) != '-'
	}
	return f.count == phDone
}
