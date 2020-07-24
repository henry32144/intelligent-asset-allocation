import React from 'react';
import Button from '@material-ui/core/Button';
import ClickAwayListener from '@material-ui/core/ClickAwayListener';
import Grow from '@material-ui/core/Grow';
import Paper from '@material-ui/core/Paper';
import Popper from '@material-ui/core/Popper';
import MenuItem from '@material-ui/core/MenuItem';
import MenuList from '@material-ui/core/MenuList';
import { makeStyles } from '@material-ui/core/styles';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import Cookies from 'universal-cookie';

const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex',

  },
  expandButton: {
    color: 'inherit'
  },
  popperRoot: {
    zIndex: 1400,
  }
}));

export default function SelectPortfolioButton(props) {
  const classes = useStyles();
  const [open, setOpen] = React.useState(false);
  const anchorRef = React.useRef(null);
  const cookies = new Cookies();

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };

  const handleClose = (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }
    setOpen(false);
  };

  const createPortfolioOnClick = (e) => {

    handleClose(e)
  };

  function handleListKeyDown(event) {
    if (event.key === 'Tab') {
      event.preventDefault();
      setOpen(false);
    }
  }

  // return focus to the button when we transitioned from !open -> open
  const prevOpen = React.useRef(open);
  React.useEffect(() => {
    if (prevOpen.current === true && open === false) {
      anchorRef.current.focus();
    }
    prevOpen.current = open;
  }, [open]);

  const userPortfolios = [{
    "protfolioId": 0,
    "portfolioName": "Portfolio 1",
    "portfolioStockIds": [0, 1, 2],
  },
  {
    "protfolioId": 1,
    "portfolioName": "Portfolio 2",
    "portfolioStockIds": [3, 4, 5],
  }
  ];

  const portfolioMenuItems = userPortfolios.map((portfolio) =>
    <MenuItem key={portfolio.protfolioId.toString()} onClick={handleClose}>
      {portfolio.portfolioName}
    </MenuItem>
  );

  return (
    <div className={classes.root}>
      <div>
        <Button
          endIcon={<ExpandMoreIcon />}
          ref={anchorRef}
          aria-controls={open ? 'menu-list-grow' : undefined}
          aria-haspopup="true"
          className={classes.expandButton}
          onClick={handleToggle}
        >
          My Portfolio
        </Button>
        <Popper 
          className={classes.popperRoot}
          open={open} 
          anchorEl={anchorRef.current} 
          role={undefined} 
          transition 
          disablePortal>
          {({ TransitionProps, placement }) => (
            <Grow
              {...TransitionProps}
              style={{ transformOrigin: placement === 'bottom' ? 'center top' : 'center bottom' }}
            >
              <Paper>
                <ClickAwayListener onClickAway={handleClose}>
                  <MenuList autoFocusItem={open} id="menu-list-grow" onKeyDown={handleListKeyDown}>
                    {portfolioMenuItems}
                    <MenuItem onClick={createPortfolioOnClick}>Create New Portfolio</MenuItem>
                  </MenuList>
                </ClickAwayListener>
              </Paper>
            </Grow>
          )}
        </Popper>
      </div>
    </div>
  );
}
